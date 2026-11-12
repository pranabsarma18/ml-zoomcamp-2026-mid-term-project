import requests

url = 'http://localhost:9696/predict'

student = {
  "id": 100616,
  "gender": "male",
  "age": 20.0,
  "city": "mumbai",
  "profession": "student",
  "academic_pressure": 2.0,
  "work_pressure": 0.0,
  "cgpa": 7.27,
  "study_satisfaction": 3.0,
  "job_satisfaction": 0.0,
  "sleep_duration": "6-7_hours",
  "dietary_habits": "healthy",
  "degree": "class_12",
  "work_or_study_hours": 9.0,
  "financial_stress": 3.0,
  "family_history_of_mental_illness": "no"
}

response = requests.post(url, json=student)
predictions = response.json()

print(predictions)
if predictions['depression_prediction']:
    print('Student seems to be depressed, send help')
else:
    print('Student is not likely to be depressed')