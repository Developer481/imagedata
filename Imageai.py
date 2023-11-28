import pygsheets
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from PIL import Image
import numpy as np
import io
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import streamlit as st
import time  # Added import for time

# Path to your credentials JSON file
credentials_file = 'cre.json'

# Authenticate using the credentials file
gc = pygsheets.authorize(service_file=credentials_file)

# Open the Google Spreadsheet using its title
spreadsheet = gc.open('Imagedatalearn')
wc = spreadsheet.sheet1

# Load data from Google Sheets
data = wc.get_all_values()
df = pd.DataFrame(data[1:], columns=data[0])

# Convert the 'Image pixel' column to a list of lists of integers
df['pixel'] = df['Image pixel'].apply(lambda x: list(map(int, x.split())))

# Filter out rows with empty 'Image pixel' values
df = df[df['pixel'].apply(lambda x: len(x) > 0)]

# Extract features and labels
X = np.array(df['pixel'].tolist())
y = LabelEncoder().fit_transform(df['Image problem Label'])

# Create and train a model
def train_model(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a model (Support Vector Classifier in this example)
    model = SVC()

    # Train the model
    model.fit(X_train, y_train)
    return model

# Initialize the model outside the button block
model = None

#predict google sheet data
def presheet(model, label_encoder):
    # Get the first column (column A) in the worksheet
    column_values = wc.get_col(1, include_empty=False)

    # Find the index of the last non-empty cell
    last_non_empty_cell_index = next(
        (idx for idx, val in enumerate(column_values[::-1], 1) if val), None)

    # Calculate the row number of the last non-empty cell
    if last_non_empty_cell_index is not None:
        ltm = len(column_values) - last_non_empty_cell_index + 1

    for i in range(2, ltm + 1):
        # Get the pixel values from the DataFrame
        cellvalue = df['pixel'][i - 2]

        # Convert the string of pixel values to an array of integers
        datapre = np.array(list(map(int, cellvalue)))

        # Predict the label using the model and update the Google Sheet
        datapre1 = predict_label(datapre, model, label_encoder)
        wc.update_value(f'B{i}', datapre1)

# Predict function
def predict_label(pixels_array, model, label_encoder):
    prediction = model.predict(pixels_array.reshape(1, -1))
    predicted_label = label_encoder.inverse_transform(prediction)
    return predicted_label[0]

# Predict labels for all images in a folder
def predict_for_all_images(model, label_encoder, folder_id):
    # Create a Drive API service object
    drive_service = build('drive', 'v3', credentials=service_account.Credentials.from_service_account_file(credentials_file))

    # List files in the specified folder
    results = drive_service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id, name)"
    ).execute()

    # Resize image function
    def resize_image(image, target_size):
        return image.resize(target_size, Image.ANTIALIAS)

    # Iterate through the files in the specified folder
    for file in results.get('files', []):
        file_id = file['id']
        file_name = file['name']

        # Download the image file
        request = drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()

        # Read pixel data from the resized image
        image = Image.open(fh)
        target_size = (16, 16)  # Adjust the target size as needed
        resized_image = resize_image(image, target_size)
        pixels_array = np.array(resized_image).flatten()

        predicted_label = predict_label(pixels_array, model, label_encoder)

        # Yield the results for each image
        yield pixels_array, file_name, predicted_label

# Update Google Sheet with predictions
def update_google_sheet(pixels_array, file_name, label, worksheet):
    try:
        # Get the first column (column A) in the worksheet
        column_values = worksheet.get_col(1, include_empty=False)

        # Find the index of the last non-empty cell
        last_non_empty_cell_index = next(
            (idx for idx, val in enumerate(column_values[::-1], 1) if val), None)

        # Calculate the row number of the last non-empty cell
        if last_non_empty_cell_index is not None:
            lsn = len(column_values) - last_non_empty_cell_index + 1
            lsn = lsn + 1

        # Update the worksheet with the values
        # Use the update_value method on the Cell object
        worksheet.update_value(f'A{lsn}', label)
        worksheet.update_value(f'B{lsn}', label)
        strvalue = ' '.join(map(str, pixels_array.tolist()))
        worksheet.update_value(f'C{lsn}', strvalue)

    except HttpError as e:
        if e.resp.status == 429:
            # Quota exceeded error, wait for some time and retry
            st.warning("Quota exceeded. Waiting for 1 minute before retrying.")
            time.sleep(60)  # Wait for 1 minute
            update_google_sheet(pixels_array, file_name, label, worksheet)
        else:
            # Handle other HttpErrors
            raise

def run():
    st.title("Image Prediction App")

    # Button to run code 1
    if st.button("Upload data for ml model"):
        # Authenticate using the credentials file
        gc = pygsheets.authorize(service_file=credentials_file)

        # Open the Google Spreadsheet using its title
        spreadsheet = gc.open('Imagedatalearn')

        # Specify the folder ID for which you want to list files
        #folder_id = st.text_input("Please enter the drive folder ID: ")
        t#ime.sleep(15)
        folder_id = "1YBhiTUQVCPZc6pcAo0PjTBnsDk7br6D4"

        # Create a Drive API service object
        drive_service = build('drive', 'v3', credentials=service_account.Credentials.from_service_account_file(credentials_file))

        # Get folder information
        folder_info = drive_service.files().get(fileId=folder_id, fields='name').execute()
        folder_name = folder_info['name']
        st.write(f"Folder Name: {folder_name}")

        # List files in the specified folder
        results = drive_service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="files(id, name)"
        ).execute()

        # Get the worksheet
        sheet_title = 'Imagedatalearn'
        worksheet = spreadsheet.sheet1

        # Get the first column (column A) in the worksheet
        column_values = worksheet.get_col(1, include_empty=False)

        # Find the index of the last non-empty cell
        last_non_empty_cell_index = next(
            (idx for idx, val in enumerate(column_values[::-1], 1) if val), None)

        # Calculate the row number of the last non-empty cell
        if last_non_empty_cell_index is not None:
            c = len(column_values) - last_non_empty_cell_index + 1
            c = c + 1

        # Resize image function
        def resize_image(image, target_size):
            return image.resize(target_size, Image.ANTIALIAS)

        # Iterate through the files in the specified folder
        for file in results.get('files', []):
            file_id = file['id']
            file_name = file['name']

            # Download the image file
            request = drive_service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()

            # Read pixel data from the resized image
            image = Image.open(fh)
            target_size = (16, 16)  # Adjust the target size as needed
            resized_image = resize_image(image, target_size)
            pixels_array = np.array(resized_image).flatten()
            pixels_string = ' '.join(map(str, pixels_array.tolist()))

            # Update the cell with the string representation of the array
            worksheet.update_value(f'C{c}', pixels_string)
            worksheet.update_value(f'A{c}', folder_name)

            c += 1

        st.success(f"Pixel data from the folder '{folder_name}' added to the Google Sheet.")

    if st.button("predict image label"):
        global model  # Declare model as a global variable
        model = train_model(X, y)
        label_encoder = LabelEncoder().fit(df['Image problem Label'])

        folder_id = "1I7m8YfThMTvBiGujEMROrg_Hm9XsgDWE"
        #folder_id = st.text_input("Please enter the drive folder ID: ")
        #time.sleep(15)
        # Predict and update for all images in the specified folder
        for pixels_array, file_name, predicted_label in predict_for_all_images(model, label_encoder, folder_id):
            st.write(f"Image: {file_name}, Predicted Label: {predicted_label}")
            update_google_sheet(pixels_array, file_name, predicted_label, wc)

    # predict google sheet data
    if model is not None:
        presheet(model, label_encoder)


if __name__ == "__main__":
    run()
