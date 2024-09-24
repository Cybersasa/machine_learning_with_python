# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 09:20:57 2024

@author: Cybersasa
"""

code into a function 
# Import the ChatGPT library from chatgpt.chat import Chat 
# Define a function that takes in the Linkedin profile as the input 
def send_message(linkedin_profile): 
    # Prompt the user to enter the name of the connection they want to message 
    connection = input("Enter the name of the connection you want to message: ") 
    # Use the ChatGPT library to generate a context for the conversation 
    context = Chat(first_speaker=linkedin_profile, second_speaker=connection) 
    # Prompt the user to enter the type of message they want to send 
    message_type = input("Choose the type of message you want to send (introduction, networking, job opportunity): ")
# Use conditional statements to generate a specific response based on the selected message type 
if message_type == "introduction": 
    # Generate an introduction template with the connection's name and common professional interests 
    introduction_template = "Hi {}, I saw your profile on LinkedIn and noticed we have common interests in {}. I would like to connect with you and learn more about your work. Best, {}".format(connection, context.common_interests(), linkedin_profile) # Prompt the user to further customize the message or use the generated template message = input("Do you want to customize the message? (y/n): ") if message == "y": message = input("Enter your customized message: ") elif message_type == "networking": # Generate a message asking for a potential collaboration or exchanging professional insights networking_message = "Hi {}, I noticed you work in a similar field and I would love to collaborate or exchange professional insights with you. Best, {}".format(connection, linkedin_profile) # Prompt the user to further customize the message or use the generated template message = input("Do you want to customize the message? (y/n): ") if message == "y": message = input("Enter your customized message: ") elif message_type == "job opportunity": # Generate a personalized message highlighting the relevant skills and experiences of the connection job_message = "Hi {}, I saw your profile on LinkedIn and noticed your experience in {} aligns with a job opportunity at my company. I would love to discuss this opportunity further with you. Best, {}".format(connection, context.experience(), linkedin_profile) # Prompt the user to further customize the message or use the generated template message = input("Do you want to customize the message? (y/n): ") if message == "y": message = input("Enter your customized message: ") else: print("Invalid message type.") # Prompt the user to enter the connection's email or username to send the message receiver = input("Enter the email or username of the connection you want to send the message to: ") # Use the Linkedin API to send the message try: send_message = context.send_message(message, receiver) print("Message successfully sent to {}!".format(connection)) except: print("Error occurred while sending the message.") # Prompt the user if they want to send another message or exit the program another_message = input("Do you want to send another message? (y/n): ") if another_message == "y": send_message(linkedin_profile) else: print("Program ended.") # Call the function and pass in the Linkedin profile as the input send_message("Your Linkedin profile")


