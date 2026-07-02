
Add a LocalProvider/EngineProvider (wrapping rullama-engine) as a follow-up — it's a natural follow-on to removing the dead hosted relay and makes the CLI's local story first-class. It's
  a clean, self-contained addition; I'd do it as its own change, not fold it into the Brainwires removal.

brainwires-framework repo was renamed: https://github.com/Brainwires/rullama-framework

We need to refactor the Deno SDK too.

Do a review both the rullama and rullama-framework projects, with a focus on what's missing in general.

AI tools allow an AI surgical editing of its context.


* If it doesn't already... Adjust so the chat history can be navigated without stopping the generation for a particulair chat. I this this works, but double check the logic is there. This way, users can review previous messages while the model is still generating a response. 

* When a new chat is created a new empty chat history entry should be added to the list... Right now a new chat is only added to the list after the first response (I think, at least after the first user message). The very first startup will need to create the first empty chat history entry as well.

* When generation is running for a chat, and the user switches to another or new chat... Any other chat should be able to queue a new message without interrupting the generation of the current chat. This way, users can seamlessly switch between different conversations and continue engaging with the model without any disruptions. The system should be designed to handle multiple chats simultaneously (running them serially in the order queued), allowing users to manage their interactions efficiently while still receiving timely responses from the model.
