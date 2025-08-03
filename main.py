from agents import Agent, AsyncOpenAI, Runner, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool
import os
from dotenv import load_dotenv
import json

load_dotenv()

set_tracing_disabled(disabled=True)





@function_tool
def get_user_info(user_id: int) -> dict:
    """
    get user info from the database and return it as a dictionary.
    """
    try:
        with open("db/user_info_test.txt", "r") as file:
            data = json.load(file)
        
        # Search for the user by ID
        for user in data["users"]:
            if user["id"] == user_id:
                return user
        
        # Return empty dict if user not found
        return {"error": f"User with ID {user_id} not found"}
    
    except FileNotFoundError:
        return {"error": "Database file not found"}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON format: {str(e)}"}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
external_client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("Gemini_API_KEY"),

)

model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-2.5-flash",
)

agent = Agent(
    name="translator",
    # description="A translator agent that translates text from english to german.",
    instructions="""
    you are a german language trainer, you will provide a practise exercise to the user based on his language score and difficulty
    level. use tool get_user_info to get the user info from the database.
    you will provide the user with a practise exercise based on his language score and difficulty level.
    if user is inactive then instead of providing a practise exercise, you will provide a irony message to user and nudge him.
    """,
    # external_client=external_client
    model=model,
    tools=[get_user_info],
)





if __name__ == "__main__":
    # user = get_user_info("1")  # Changed to existing user ID
    # print(user)
    response = Runner.run_sync(agent, "suggest me todays german exercise for user with id 1")
    print(response.final_output)
