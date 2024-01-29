from django.http import HttpResponse
from django.shortcuts import render
import joblib

def home(request):
    return render(request,'home.html')

''' 
def result(request):
    ada_model = joblib.load('finalizedModel.sav')
    lis = []
    lis.append(request.GET['gender'])
    lis.append(request.GET['age'])
    lis.append(request.GET['hypertension'])
    lis.append(request.GET['heart_disease'])
    lis.append(request.GET['smoking_history'])
    lis.append(request.GET['bmi'])
    lis.append(request.GET['HbA1c_level'])
    lis.append(request.GET['blood_glucose_level'])
     
    print(lis)
    
    return render(request,'result.html')
    #return HttpResponse('This is homepage')'''
def result(request):
    if request.method == 'POST':  # Check if the request method is POST
        ada_model = joblib.load('finalizedModel.sav')
        
        # Retrieve form data from request.POST
        gender = float(request.POST.get('gender'))  # Convert to float if necessary
        age = float(request.POST.get('age'))
        hypertension = float(request.POST.get('hypertension'))
        heart_disease = float(request.POST.get('heart_disease'))
        smoking_history = float(request.POST.get('smoking_history'))
        bmi = float(request.POST.get('bmi'))
        HbA1c_level = float(request.POST.get('HbA1c_level'))
        blood_glucose_level = float(request.POST.get('blood_glucose_level'))
        
        # Make prediction using the model
        input_data = [[gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]]
        prediction = ada_model.predict(input_data)
        
        # Map prediction to a human-readable format if needed
        
        # Pass input data and prediction result as context to the template
        context = {
            'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'smoking_history': smoking_history,
            'bmi': bmi,
            'HbA1c_level': HbA1c_level,
            'blood_glucose_level': blood_glucose_level,
            'prediction': prediction[0]  # Assuming prediction is a single value
        }
        
        # Render the result.html template with the context
        #return render(request, 'result.html', {'ans': prediction})
        return render(request, 'result.html', context)
    else:
        # Handle cases where request method is not POST, maybe display an error message or redirect
        return HttpResponse('Invalid request method')