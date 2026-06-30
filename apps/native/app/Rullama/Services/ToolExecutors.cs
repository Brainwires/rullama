using System;
using System.Globalization;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading.Tasks;

namespace Rullama.Services;

/// <summary>
/// Executes tool calls. Weather via WeatherAPI.com (if a key is set) else the
/// free, keyless Open-Meteo. Location falls back to IP geolocation when the
/// model omits it and the user opted in. Other tools return a stub for now.
/// </summary>
public sealed class ToolExecutors
{
    private static readonly HttpClient Http = new() { Timeout = TimeSpan.FromSeconds(15) };

    public bool UseFahrenheit { get; init; }
    public string? WeatherApiKey { get; init; }
    public bool UseLocation { get; init; }

    public async Task<string> ExecuteAsync(ToolCall call)
    {
        try
        {
            return call.Name switch
            {
                "get_weather" => await WeatherAsync(Arg(call, "location"), days: 0),
                "get_weather_forecast" => await WeatherAsync(Arg(call, "location"), days: ParseDays(Arg(call, "days"))),
                _ => $"(tool '{call.Name}' is not available yet)",
            };
        }
        catch (Exception e)
        {
            return $"(error running {call.Name}: {e.Message})";
        }
    }

    private static string Arg(ToolCall c, string key) => c.Args.TryGetValue(key, out string? v) ? v : "";
    private static int ParseDays(string s) => int.TryParse(s, out int d) ? Math.Clamp(d, 1, 10) : 3;

    private async Task<string> WeatherAsync(string location, int days)
    {
        string unitsLabel = UseFahrenheit ? "°F" : "°C";

        // Resolve a place + coordinates.
        (double lat, double lon, string place) = await ResolveLocationAsync(location);

        if (!string.IsNullOrWhiteSpace(WeatherApiKey))
            return await WeatherApiAsync(location.Length > 0 ? location : $"{lat},{lon}", place, days);

        string tempUnit = UseFahrenheit ? "fahrenheit" : "celsius";
        if (days <= 0)
        {
            string url = $"https://api.open-meteo.com/v1/forecast?latitude={lat.ToString(CultureInfo.InvariantCulture)}&longitude={lon.ToString(CultureInfo.InvariantCulture)}&current=temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m&temperature_unit={tempUnit}";
            using JsonDocument doc = await GetJsonAsync(url);
            JsonElement cur = doc.RootElement.GetProperty("current");
            double temp = cur.GetProperty("temperature_2m").GetDouble();
            int code = cur.GetProperty("weather_code").GetInt32();
            double hum = cur.GetProperty("relative_humidity_2m").GetDouble();
            return $"{place}: {Wmo(code)}, {temp:0.#}{unitsLabel}, humidity {hum:0}%.";
        }
        else
        {
            string url = $"https://api.open-meteo.com/v1/forecast?latitude={lat.ToString(CultureInfo.InvariantCulture)}&longitude={lon.ToString(CultureInfo.InvariantCulture)}&daily=temperature_2m_max,temperature_2m_min,weather_code&forecast_days={days}&temperature_unit={tempUnit}";
            using JsonDocument doc = await GetJsonAsync(url);
            JsonElement daily = doc.RootElement.GetProperty("daily");
            JsonElement time = daily.GetProperty("time");
            JsonElement hi = daily.GetProperty("temperature_2m_max");
            JsonElement lo = daily.GetProperty("temperature_2m_min");
            JsonElement codes = daily.GetProperty("weather_code");
            var sb = new System.Text.StringBuilder($"{place} forecast:\n");
            for (int i = 0; i < time.GetArrayLength(); i++)
                sb.Append($"- {time[i].GetString()}: {Wmo(codes[i].GetInt32())}, {lo[i].GetDouble():0.#}–{hi[i].GetDouble():0.#}{unitsLabel}\n");
            return sb.ToString().TrimEnd();
        }
    }

    private async Task<string> WeatherApiAsync(string q, string place, int days)
    {
        if (days <= 0)
        {
            string url = $"https://api.weatherapi.com/v1/current.json?key={WeatherApiKey}&q={Uri.EscapeDataString(q)}";
            using JsonDocument doc = await GetJsonAsync(url);
            JsonElement cur = doc.RootElement.GetProperty("current");
            double temp = UseFahrenheit ? cur.GetProperty("temp_f").GetDouble() : cur.GetProperty("temp_c").GetDouble();
            string cond = cur.GetProperty("condition").GetProperty("text").GetString() ?? "";
            return $"{place}: {cond}, {temp:0.#}{(UseFahrenheit ? "°F" : "°C")}.";
        }
        string furl = $"https://api.weatherapi.com/v1/forecast.json?key={WeatherApiKey}&q={Uri.EscapeDataString(q)}&days={days}";
        using JsonDocument fdoc = await GetJsonAsync(furl);
        var sb = new System.Text.StringBuilder($"{place} forecast:\n");
        foreach (JsonElement day in fdoc.RootElement.GetProperty("forecast").GetProperty("forecastday").EnumerateArray())
        {
            JsonElement d = day.GetProperty("day");
            string date = day.GetProperty("date").GetString() ?? "";
            double hi = UseFahrenheit ? d.GetProperty("maxtemp_f").GetDouble() : d.GetProperty("maxtemp_c").GetDouble();
            double lo = UseFahrenheit ? d.GetProperty("mintemp_f").GetDouble() : d.GetProperty("mintemp_c").GetDouble();
            string cond = d.GetProperty("condition").GetProperty("text").GetString() ?? "";
            sb.Append($"- {date}: {cond}, {lo:0.#}–{hi:0.#}{(UseFahrenheit ? "°F" : "°C")}\n");
        }
        return sb.ToString().TrimEnd();
    }

    private async Task<(double Lat, double Lon, string Place)> ResolveLocationAsync(string location)
    {
        location = location.Trim();
        // "lat,lon"
        string[] parts = location.Split(',');
        if (parts.Length == 2
            && double.TryParse(parts[0], NumberStyles.Float, CultureInfo.InvariantCulture, out double la)
            && double.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out double lo))
            return (la, lo, location);

        if (location.Length == 0)
        {
            if (!UseLocation) throw new InvalidOperationException("no location given");
            using JsonDocument ip = await GetJsonAsync("http://ip-api.com/json");
            return (ip.RootElement.GetProperty("lat").GetDouble(),
                    ip.RootElement.GetProperty("lon").GetDouble(),
                    ip.RootElement.TryGetProperty("city", out JsonElement c) ? c.GetString() ?? "your location" : "your location");
        }

        string url = $"https://geocoding-api.open-meteo.com/v1/search?name={Uri.EscapeDataString(location)}&count=1";
        using JsonDocument doc = await GetJsonAsync(url);
        if (!doc.RootElement.TryGetProperty("results", out JsonElement results) || results.GetArrayLength() == 0)
            throw new InvalidOperationException($"could not find '{location}'");
        JsonElement r = results[0];
        string name = r.GetProperty("name").GetString() ?? location;
        string country = r.TryGetProperty("country", out JsonElement co) ? ", " + co.GetString() : "";
        return (r.GetProperty("latitude").GetDouble(), r.GetProperty("longitude").GetDouble(), name + country);
    }

    private static async Task<JsonDocument> GetJsonAsync(string url)
    {
        using HttpResponseMessage resp = await Http.GetAsync(url);
        resp.EnsureSuccessStatusCode();
        return await JsonDocument.ParseAsync(await resp.Content.ReadAsStreamAsync());
    }

    private static string Wmo(int code) => code switch
    {
        0 => "clear sky",
        1 or 2 or 3 => "partly cloudy",
        45 or 48 => "fog",
        51 or 53 or 55 => "drizzle",
        61 or 63 or 65 => "rain",
        66 or 67 => "freezing rain",
        71 or 73 or 75 or 77 => "snow",
        80 or 81 or 82 => "rain showers",
        85 or 86 => "snow showers",
        95 => "thunderstorm",
        96 or 99 => "thunderstorm with hail",
        _ => "unknown",
    };
}
