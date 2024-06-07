import streamlit as st
import pandas as pd
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from time import sleep
import os
import nltk
nltk.download('punkt')

# Function to preprocess text
def preprocess_text(text):
    if pd.isna(text): 
        return ''
    else:
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token not in stopwords.words('english')]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        cleaned_text = ' '.join(tokens)
        return cleaned_text

# Function to scrape Facebook comments and save them to CSV
def scrape_comments():
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    driver.get("https://www.facebook.com")
    driver.maximize_window()
    sleep(2)

    try:
        cookies = WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Accept')]")))
        cookies.click()
        st.write("Cookies accepted")
    except TimeoutException:
        st.write("Cookie button not found or not clickable")

    email = driver.find_element(By.ID, "email")
    email.send_keys("57256604")
    password = driver.find_element(By.ID, "pass")
    password.send_keys("reuben18@.")
    login_button = driver.find_element(By.NAME, "login")
    login_button.click()
    specific_page_url = "https://www.facebook.com/MejametalanaNewsroom"
    driver.get(specific_page_url)
    sleep(2)

    comment_list = []
    scroll_count = 5
    scroll_delay = 2

    for _ in range(scroll_count):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        sleep(scroll_delay)
        comments = driver.find_elements(By.XPATH, "//div[@class='x1lliihq xjkvuk6 x1iorvi4']")
        for comment in comments:
            comment_list.append(comment.text.strip())

    df_comments = pd.DataFrame({"Comments": comment_list})
    
    # Update this path to a directory where you have write permissions
    output_path = os.path.expanduser("~/Desktop/IT IS WELL/Project/Final/facebook_comments.csv")
    df_comments.to_csv(output_path, index=False)
    st.subheader("Scraped Data")
    st.dataframe(df_comments)
    perform_analysis(df_comments)

# Function to perform analysis on the scraped data
def perform_analysis(df_comments):
    df_comments['Cleaned_Comment'] = df_comments['Comments'].apply(preprocess_text)
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    loaded_model = joblib.load('random_forest_model.joblib')
    X_comments = vectorizer.transform(df_comments['Cleaned_Comment']).toarray()
    df_comments['bullying_type'] = loaded_model.predict(X_comments)
    st.subheader("Analyzed Data")
    st.dataframe(df_comments)

st.title("Facebook Comments Scraper, Data Cleaning & Detection Application")
st.markdown(
    """
    <style>
        .stAlert {
            padding: 10px;
            background-color: #f44336;
            color: white;
            border-radius: 5px;
            margin-top: 10px;
        }
        .dataframe {
            border: 2px dashed #ddd;
            border-collapse: collapse;
            margin-top: 10px;
        }
        .dataframe th, .dataframe td {
            border: 2px solid #ddd;
            padding: 8px;
            text-align: left;
            color: red;
        }
    </style>
    """,
    unsafe_allow_html=True
)

nav_choice = st.sidebar.radio("", ["Scraping", "Analysis", "Detection"])
df_comments = pd.DataFrame({"Comments": []})

if nav_choice == "Scraping":
    if st.button("Scrape and Save Comments"):
        scrape_comments()
        st.success("Comments scraped, saved to CSV, and analyzed!")

elif nav_choice == "Analysis":
    if st.button("Analyze Comments"):
        df_comments = pd.read_csv("facebook_comments.csv")
        perform_analysis(df_comments)
        st.success("Comments analyzed!")

elif nav_choice == "Detection":
    text_input = st.text_input("Enter text for bullying detection:")
    if text_input:
        cleaned_bullying_words = preprocess_text(text_input)
        vectorizer = joblib.load('tfidf_vectorizer.joblib')
        loaded_model = joblib.load('random_forest_model.joblib')
        X_input = vectorizer.transform([cleaned_bullying_words]).toarray()
        predicted_type = loaded_model.predict(X_input)[0]
        class_name = "Bully" if predicted_type == 1 else "Non-Bully"
        #printing result
        st.subheader("RESULTS")
        st.info(f"Predicted Bullying Type: {class_name}")
        st.write(f"Predicted Bullying Type: {predicted_type}")