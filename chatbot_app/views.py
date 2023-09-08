# from rest_framework.decorators import api_view
# from rest_framework.response import Response
# import nltk
# from django.http import HttpResponse, JsonResponse
# # from django.views.decorators.csrf import csrf_exempt
# from django.http import HttpResponse
# import json
# from django.http import JsonResponse
# from django.views.decorators.http import require_POST
from .model_logic import preparation, load_response, generate_response


# initialize the lemmatizer and stopwords
# lemmatizer = WordNetLemmatizer()
# stop_words = set(stopwords.words('english'))
from django.http import request
from rest_framework.decorators import api_view
from rest_framework.response import Response
import json
from django.http import HttpResponse, JsonResponse
# from django.views.decorators.csrf import csrf_exempt
from .model_logic import generate_response
preparation()
load_response()
# @csrf_exempt
def chatbot(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        text = data.get('message')
        response = generate_response(text)
        message = {"answer": response}
        return JsonResponse(message)
    else:
        return HttpResponse("Invalid request method")

# @api_view(['POST'])
# def chatbot(request):
#     # get the user's message from the POST request
#     message = request.data['message']
#     # tokenize the user's message
#     words = nltk.word_tokenize(message.lower())
#     # remove stop words from the user's message
#     words = [word for word in words if word not in stop_words]
#     # lemmatize the remaining words in the user's message
#     words = [lemmatizer.lemmatize(word) for word in words]
#     # determine the chatbot's response based on the user's message
#     response = 'Hello, how can I help you?'
#     if 'help' in words:
#         response = generate_response(message)
#     elif 'problem' in words:
#         response = 'What seems to be the problem?'
#     elif 'thanks' in words or 'thank you' in words:
#         response = 'You\'re welcome!'
#     # return the chatbot's response in a JSON format
#     return Response({'message': response})
# def chatbot(request):
#       if request.method == 'POST':
#         data = json.loads(request.body)
#         text = data.get('message')
#         response = generate_response(text)  # Your generate_response function implementation
#         message = {"answer": response}
#         return HttpResponse(json.dumps(message), content_type="application/json")
# def chatbot(request):
#     if request.method == 'POST':
#         data = json.loads(request.body)
#         text = data.get('message')
#         response = generate_response(text)
#         message = {"answer": response}
#         return HttpResponse(json.dumps(str(message)), content_type="application/json")
        # return JsonResponse(message)

# @require_POST
# def chatbot(request):
#     if request.method == 'POST':
#         data = request.POST.get('data')
        
#         # Perform any necessary data processing here
        
#         # Call the generate_response function with the processed data
#         response = generate_response(data)
        
#         # Prepare the response as a JSON object
#         response_data = {
#             'response': response
#         }
        
#         # Return the JSON response
#         return JsonResponse(response_data)