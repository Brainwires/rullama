//! Microbenchmark for the batched f16 matmul on vision-representative
//! shapes. Times naive, v1 tiled, and v2 tiled at the same shape so we
//! can see how much of the 51 s vision encode is actually matmul.
//!
//! The big vision matmuls are:
//!   • qkv:  k=768,  n=768,  batch=2304  (×3 q,k,v separately)
//!   • attn: k=768,  n=768,  batch=2304
//!   • ffn_up/gate:  k=768,  n=3072, batch=2304
//!   • ffn_down:     k=3072, n=768,  batch=2304
//!
//! 16 blocks of those = ~96 matmuls per encode. If each takes ~500 ms
//! the encode is matmul-bound. If each takes ~50 ms, something else
//! is eating the time.

use std::time::Instant;

use rullama::backend::{Pipelines, WgpuCtx, dispatch};

fn run_shape(
    label: &str,
    ctx: &WgpuCtx,
    pipes: &Pipelines,
    w_buf: &wgpu::Buffer,
    x_buf: &wgpu::Buffer,
    y_buf: &wgpu::Buffer,
    k: usize, n: usize, batch: usize,
    n_iters: usize,
) {
    // Warmup.
    for _ in 0..3 {
        let mut enc = ctx.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None });
        dispatch::matmul_f16_batched_chained(ctx, pipes, &mut enc, w_buf, x_buf, y_buf, k, n, batch);
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
    }

    let t = Instant::now();
    for _ in 0..n_iters {
        let mut enc = ctx.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None });
        dispatch::matmul_f16_batched_chained(ctx, pipes, &mut enc, w_buf, x_buf, y_buf, k, n, batch);
        ctx.queue.submit(Some(enc.finish()));
    }
    ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
    let elapsed = t.elapsed();
    let per_iter = elapsed / n_iters as u32;
    let gflops = 2.0 * (k * n * batch) as f64 / per_iter.as_secs_f64() / 1e9;
    println!("{label:30} k={k:5} n={n:5} batch={batch:5}: {per_iter:?}/iter   {gflops:.2} GFLOPS");
}

fn run_shape_force(
    label: &str,
    ctx: &WgpuCtx,
    pipes: &Pipelines,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    dispatch_x: u32, dispatch_y: u32,
    k: usize, n: usize, batch: usize,
    n_iters: usize,
) {
    // Warmup.
    for _ in 0..3 {
        let mut enc = ctx.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None, timestamp_writes: None,
            });
            cp.set_pipeline(pipeline);
            cp.set_bind_group(0, bind_group, &[]);
            cp.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
    }

    let t = Instant::now();
    for _ in 0..n_iters {
        let mut enc = ctx.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None, timestamp_writes: None,
            });
            cp.set_pipeline(pipeline);
            cp.set_bind_group(0, bind_group, &[]);
            cp.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
    }
    ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
    let elapsed = t.elapsed();
    let per_iter = elapsed / n_iters as u32;
    let gflops = 2.0 * (k * n * batch) as f64 / per_iter.as_secs_f64() / 1e9;
    let _ = pipes; // suppress unused
    let _ = n; let _ = batch;
    println!("{label:30}                                       : {per_iter:?}/iter   {gflops:.2} GFLOPS");
}

fn f32_to_f16_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; values.len() * 2];
    for (i, &v) in values.iter().enumerate() {
        let h = half::f16::from_f32(v).to_bits();
        out[i*2]     = (h & 0xFF) as u8;
        out[i*2 + 1] = (h >> 8)   as u8;
    }
    out
}

fn make_buffers(ctx: &WgpuCtx, k: usize, n: usize, batch: usize) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
    let mut state: u32 = 0xCAFEFACE;
    let mut next = || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        ((state >> 8) as f32 / 16777216.0) - 0.5
    };
    let w_f32: Vec<f32> = (0..n * k).map(|_| next() * 0.05).collect();
    let x: Vec<f32> = (0..batch * k).map(|_| next() * 0.5).collect();
    let w_bytes = f32_to_f16_bytes(&w_f32);

    let w_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bench.w"), size: w_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&w_buf, 0, &w_bytes);
    let x_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bench.x"), size: (x.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&x_buf, 0, bytemuck::cast_slice(&x));
    let y_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bench.y"), size: (batch * n * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    (w_buf, x_buf, y_buf)
}

fn main() {
    let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu init failed");
    let pipes = Pipelines::new(&ctx.device);
    let info = ctx.adapter.get_info();
    println!("Adapter: {} / {:?}", info.name, info.backend);

    // Real vision shapes.
    let shapes = [
        ("attn QKV  768x768",     768, 768,  2304),
        ("attn out  768x768",     768, 768,  2304),
        ("ffn up    768x3072",    768, 3072, 2304),
        ("ffn down  3072x768",   3072, 768,  2304),
    ];

    // Pre-allocate buffers for each shape (separately, since k,n,batch vary).
    println!("\n=== full router (whatever variant fires) ===");
    for (label, k, n, batch) in shapes {
        let (w, x, y) = make_buffers(&ctx, k, n, batch);
        run_shape(label, &ctx, &pipes, &w, &x, &y, k, n, batch, 5);
    }

    // Now force each variant for the biggest shape (ffn_up) so we can see
    // the spread directly.
    println!("\n=== forced variants on ffn_up shape (k=768, n=3072, batch=2304) ===");
    let k = 768;
    let n = 3072;
    let batch = 2304;
    let (w, x, y) = make_buffers(&ctx, k, n, batch);

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params { k: u32, n: u32, batch: u32, _pad: u32 }
    let params = Params { k: k as u32, n: n as u32, batch: batch as u32, _pad: 0 };
    let p_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bench.p"), size: std::mem::size_of::<Params>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&p_buf, 0, bytemuck::bytes_of(&params));

    let mk_bg = |pipeline: &wgpu::ComputePipeline| {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: y.as_entire_binding() },
            ],
        })
    };

    let bg_naive = mk_bg(&pipes.f16_matmul_batched);
    let bg_v1    = mk_bg(&pipes.f16_matmul_batched_tiled);
    let bg_v2    = mk_bg(&pipes.f16_matmul_batched_tiled_v2);

    run_shape_force("naive", &ctx, &pipes, &pipes.f16_matmul_batched,
        &bg_naive, (n as u32).div_ceil(64), batch as u32, k, n, batch, 5);
    run_shape_force("v1 tiled 8×8×16", &ctx, &pipes, &pipes.f16_matmul_batched_tiled,
        &bg_v1, (n as u32).div_ceil(8), (batch as u32).div_ceil(8), k, n, batch, 5);
    run_shape_force("v2 tiled 16×16×16", &ctx, &pipes, &pipes.f16_matmul_batched_tiled_v2,
        &bg_v2, (n as u32).div_ceil(16), (batch as u32).div_ceil(16), k, n, batch, 5);

    // Per-block estimate: 6 matmuls per block × 16 blocks = 96 matmuls.
    // Avg matmul ~ middle of the shapes above.
    println!("\n96 matmuls × per-iter from above ≈ expected total matmul time per encode.");

    // ---- vision_attention bench ----
    println!("\n=== vision_attention (n_patches=2304, n_heads=12, head_dim=64) ===");
    let n_patches = 2304usize;
    let n_heads = 12usize;
    let head_dim = 64usize;
    let qkv_size = (n_patches * n_heads * head_dim * 4) as u64;
    let mkbuf = |label| ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label), size: qkv_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let q_buf = mkbuf("attn.q");
    let k_buf = mkbuf("attn.k");
    let v_buf = mkbuf("attn.v");
    let out_buf = mkbuf("attn.out");
    let zeros = vec![0f32; n_patches * n_heads * head_dim];
    ctx.queue.write_buffer(&q_buf, 0, bytemuck::cast_slice(&zeros));
    ctx.queue.write_buffer(&k_buf, 0, bytemuck::cast_slice(&zeros));
    ctx.queue.write_buffer(&v_buf, 0, bytemuck::cast_slice(&zeros));

    // Warmup.
    for _ in 0..2 {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        dispatch::vision_attention_chained(&ctx, &pipes, &mut enc, &q_buf, &k_buf, &v_buf, &out_buf,
            head_dim, n_heads, n_patches);
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
    }
    let t = Instant::now();
    let n_iters = 5;
    for _ in 0..n_iters {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        dispatch::vision_attention_chained(&ctx, &pipes, &mut enc, &q_buf, &k_buf, &v_buf, &out_buf,
            head_dim, n_heads, n_patches);
        ctx.queue.submit(Some(enc.finish()));
    }
    ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
    let elapsed = t.elapsed();
    let per_iter = elapsed / n_iters as u32;
    println!("vision_attention            : {per_iter:?}/iter  (×16 blocks ≈ {:?} total)", per_iter * 16);
}
