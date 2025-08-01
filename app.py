from flask import Flask, render_template, request
import os
import joblib
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
import json

app = Flask(__name__)

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# ✅ Load Random Forest model + TF-IDF vectorizer
model = joblib.load("rf_spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ✅ Gmail authentication (same as before)
def authenticate_gmail(user_email):
    token_file = f"token_{user_email}.json"

    creds = None
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    else:
        creds_json = os.getenv("GOOGLE_CREDENTIALS")
        if creds_json is None:
            raise RuntimeError("GOOGLE_CREDENTIALS not set in environment.")
        with open("credentials.json", "w") as f:
            f.write(creds_json)

        flow = InstalledAppFlow.from_client_config({
    "installed": {
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
        "redirect_uris": [os.getenv("GOOGLE_REDIRECT_URI")],
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token"
    }
}, SCOPES)

        creds = flow.run_console()
        with open(token_file, 'w') as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)

# ✅ Fetch and classify recent emails
def get_recent_emails(service, num=5):
    results = service.users().messages().list(userId='me', maxResults=num).execute()
    messages = results.get('messages', [])
    emails = []

    for msg in messages:
        msg_data = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
        snippet = msg_data.get('snippet', '')
        vector = vectorizer.transform([snippet])
        prediction = model.predict(vector)[0]
        label = "🚨 Spam" if prediction == 1 else "✅ Not Spam"
        emails.append({"snippet": snippet, "result": label})

    return emails

# Home page
@app.route("/")
def index():
    return render_template("index.html", emails=None)

# Gmail check route
@app.route("/check", methods=["POST"])
def check():
    user_email = request.form.get("email")
    email_count = int(request.form.get("count", 5))

    service = authenticate_gmail(user_email)
    emails = get_recent_emails(service, num=email_count)

    return render_template("index.html", emails=emails)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

