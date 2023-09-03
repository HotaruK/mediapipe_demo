import mediapipe as mp

# Create a variable for the input text
input_text = "From a Californian: What the East Coast can expect as wildfire smoke moves in"

if __name__ == '__main__':
    model = mp.solutions.text_detection.TextDetection()
    # Get the parameters and their meanings from the model
    parameters = model.get_parameters()
    for parameter in parameters:
        print(parameter, parameters[parameter])

    # Process the input text
    processed_text = model.process(input_text)

    # Get the sentiment of the input text
    sentiment = processed_text.sentiment

    # Print the sentiment of the input text
    print("The sentiment of the input text is:", sentiment)

    # Print all the parameters with their names
    for parameter in parameters:
        print(f"{parameter}: {parameters[parameter]}")
