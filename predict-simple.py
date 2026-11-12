
#!/usr/bin/env python

import pickle
import pandas as pd


MODEL_FILE = 'model.bin'


def load_model():
    with open(MODEL_FILE, 'rb') as f_in:
        pipeline = pickle.load(f_in)

    return pipeline


def prepare_data(datapoint):
    df = pd.DataFrame([datapoint])

    # Normalize column names
    df.columns = (
        df.columns.str.lower()
        .str.replace(' ', '_')
        .str.replace('/', '_or_')
    )

    # Rename columns
    df = df.rename(columns={
        'have_you_ever_had_suicidal_thoughts_?': 'suicidal_thoughts'
    })

    # Normalize categorical values
    categorical_columns = df.select_dtypes(
        include=['object', 'string']
    ).columns

    for col in categorical_columns:
        df[col] = (
            df[col]
            .str.lower()
            .str.replace(' ', '_', regex=False)
        )

    # Convert sleep duration to numerical average
    df['sleep_duration'] = (
        df['sleep_duration']
        .astype(str)
        .str.extract(r'(\d+)(?:-(\d+))?')
        .astype(float)
        .mean(axis=1)
    )

    # Remove excluded columns
    df = df.drop(
        columns=['id', 'depression', 'suicidal_thoughts'],
        errors='ignore'
    )

    return df


def predict(datapoint, pipeline=None):

    if pipeline is None:
        pipeline = load_model()

    df = prepare_data(datapoint)

    probability = pipeline.predict_proba(df)[0, 1]
    prediction = int(probability >= 0.5)

    return {
        'depression_probability': float(probability),
        'depression_prediction': prediction
    }


if __name__ == '__main__':

    datapoint = {
        'id': 100616,
        'gender': 'female',
        'age': 20.0,
        'city': 'mumbai',
        'profession': 'student',
        'academic_pressure': 3.0,
        'work_pressure': 0.0,
        'cgpa': 7.27,
        'study_satisfaction': 3.0,
        'job_satisfaction': 0.0,
        'sleep_duration': '5-6_hours',
        'dietary_habits': 'healthy',
        'degree': 'class_12',
        'have_you_ever_had_suicidal_thoughts_?': 'yes',
        'work_or_study_hours': 7.0,
        'financial_stress': 2.0,
        'family_history_of_mental_illness': 'yes'
    }

    pipeline = load_model()

    result = predict(datapoint, pipeline)

    print(f"Depression probability: {result['depression_probability']:.3f}")
    print(f"Predicted class: {result['depression_prediction']}")
