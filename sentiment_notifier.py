import pandas as pd
import os
import datetime
import glob
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication # For attaching files
import subprocess # To run review_tools.py --update
from jinja2 import Environment, FileSystemLoader # Import Jinja2

# --- CONFIGURATION ---
# IMPORTANT: Replace with your actual email details and threshold
SENDER_EMAIL = "chikachiamaka@gmail.com"  # Your email address
SENDER_PASSWORD = os.getenv("EMAIL_APP_PASSWORD") # Use an App Password for Gmail/Outlook if 2FA is on
RECEIVER_EMAILS = ["adeiza.isiaka@yahoo.com", "isiakachiamaka@hotmail.com", "g.isiaka@myseneca.ca"] # List of recipient emails
SMTP_SERVER = "smtp.gmail.com" # e.g., "smtp.gmail.com" for Gmail, "smtp-mail.outlook.com" for Outlook
SMTP_PORT = 587

# --- Thresholds 
NEGATIVE_SPIKE_THRESHOLD = 68.0 # Percentage: e.g., alert if negative sentiment > 15%
NEGATIVE_INCREASE_PERCENTAGE = 8.0 # Percentage: e.g., alert if negative sentiment increased by 20% compared to previous


# --- Jinja2 Setup ---
# Set up Jinja2 environment to load templates from the current directory
template_loader = FileSystemLoader(searchpath=".")
jinja_env = Environment(loader=template_loader)
EMAIL_TEMPLATE = jinja_env.get_template("email_template.html")

# --- Helper Functions ---
def get_latest_processed_file():
    """Finds the latest processed review CSV file."""
    files = glob.glob("updated_reviews_*.csv")
    if not files:
        return None
    latest_file = max(files, key=os.path.getctime)
    return latest_file

def get_previous_processed_file(current_file_path):
    """Finds the second latest processed review CSV file for comparison."""
    files = sorted(glob.glob("updated_reviews_*.csv"), key=os.path.getctime, reverse=True)
    if len(files) < 2:
        return None # Not enough files for comparison

    # Filter out the current_file_path if it's explicitly in the list (though max should prevent it)
    files = [f for f in files if f != current_file_path]

    if len(files) >= 1:
        return files[0] # The second latest file
    return None


def send_email(subject, html_body, recipients, bcc_recipients=None, attachment_path=None): # Added bcc_recipients parameter
    """Sends an HTML email with the given subject, body, and recipients."""
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("Error: SENDER_EMAIL or SENDER_PASSWORD environment variables not set. Cannot send email.")
        return

    try:
        msg = MIMEMultipart("alternative")
        msg['From'] = SENDER_EMAIL
        msg['To'] = ", ".join(recipients) # These will be visible
        msg['Subject'] = subject

        if bcc_recipients: # Add BCC header if bcc_recipients are provided
            msg['Bcc'] = ", ".join(bcc_recipients)

        msg.attach(MIMEText(html_body, 'html'))

        if attachment_path:
            with open(attachment_path, "rb") as f:
                attach = MIMEApplication(f.read(), _subtype="csv")
                attach.add_header('Content-Disposition', 'attachment', filename=os.path.basename(attachment_path))
                msg.attach(attach)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            # The send_message method automatically handles To, Cc, and Bcc from the msg object
            server.send_message(msg)
        print(f"Email sent successfully to {', '.join(recipients)}")
        if bcc_recipients:
            print(f"BCC'd to {', '.join(bcc_recipients)}")
    except Exception as e:
        print(f"Failed to send email: {e}")
        print("Please check your email configuration (SENDER_EMAIL, SENDER_PASSWORD, SMTP_SERVER, SMTP_PORT) and app password if using Gmail/Outlook with 2FA.")

def generate_sentiment_report_and_alerts():
    print("--- Starting Sentiment Notifier ---")

    # Define your main recipients (e.g., team lead, yourself) and BCC recipients
    main_recipients = ["chikachiamaka@gmail.com"] # The primary person/group you want in 'To'
    bcc_stakeholders = ["adeiza.isiaka@yahoo.com", "isiakachiamaka@hotmail.com", "g.isiaka@myseneca.ca"] # Your list of BCC'd stakeholders

    # 1. Run review_tools.py --update to get the latest data
    print("Running review_tools.py --update to get fresh data...")
    try:
        my_env = os.environ.copy()
        my_env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run(
            ["python", "review_tools.py", "--update"],
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8', # Explicitly tell it to decode as UTF-8
            env=my_env
        )
        print("review_tools.py output:\n", result.stdout)
        if result.stderr:
            print("review_tools.py errors:\n", result.stderr)
        print("Data update complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error running review_tools.py --update: {e}")
        print(f"Stderr: {e.stderr}")
        send_email(
            "Sentiment Notifier Error",
            f"Failed to update reviews. Please check the server logs.\nError: {e.stderr}",
            # Use main_recipients here if this error email should only go to the "To" group,
            # or use RECEIVER_EMAILS if you want everyone to get it (as it was before).
            # For simplicity, I'll keep RECEIVER_EMAILS here if that's what you prefer for error notifications.
            RECEIVER_EMAILS
        )
        return
    
    # 2. Load latest processed data
    current_file = get_latest_processed_file()
    if not current_file:
        print("No processed review files found after update. Exiting.")
        send_email(
            "Sentiment Notifier - No Data Found",
            "No processed review files were found after running the update script.",
            RECEIVER_EMAILS
        )
        return

    print(f"Loading data from {current_file}")
    df = pd.read_csv(current_file, parse_dates=["at"])

    subject = f"Shein Review Sentiment Report - {datetime.date.today().strftime('%Y-%m-%d')}"
    
    # --- Prepare Data for Template ---
    template_data = {
        "current_date": datetime.date.today().strftime('%Y-%m-%d'),
        "data_source": os.path.basename(current_file),
        "total_reviews": len(df),
        "overall_sentiment_dist": {}, # Will be populated below
        "sentiment_by_theme": {}, # Will be populated below
        "sentiment_by_theme_cols": ['NEGATIVE', 'NEUTRAL', 'POSITIVE'], # For consistent column order
        "alerts": [] # List of dictionaries: {'type': 'alert-red', 'message': '...'}
    }

    # Overall Sentiment Distribution
    overall_sentiment_dist = df['sentiment'].value_counts(normalize=True) * 100
    template_data["overall_sentiment_dist"] = overall_sentiment_dist.round(2).to_dict()

    # Sentiment Distribution by Theme
    sentiment_by_theme_df = pd.pivot_table(df, index='theme', columns='sentiment', aggfunc='size', fill_value=0)
    for col in template_data["sentiment_by_theme_cols"]:
        if col not in sentiment_by_theme_df.columns:
            sentiment_by_theme_df[col] = 0
    sentiment_by_theme_df = sentiment_by_theme_df[template_data["sentiment_by_theme_cols"]] # Order them
    template_data["sentiment_by_theme"] = sentiment_by_theme_df.to_dict(orient='index')

    # --- Negative Sentiment Spike Alert ---
    current_negative_sentiment_pct = overall_sentiment_dist.get('NEGATIVE', 0.0)
    
    # 1. Threshold Alert
    if current_negative_sentiment_pct > NEGATIVE_SPIKE_THRESHOLD:
        template_data["alerts"].append({
            'type': 'alert-red',
            'message': f"🚨 CRITICAL ALERT: Negative sentiment is currently at {current_negative_sentiment_pct:.2f}% (above threshold of {NEGATIVE_SPIKE_THRESHOLD}%)"
        })
        subject = "🚨 ALERT: Shein Review Sentiment Spike!" # Update subject for alerts

    # 2. Comparison to Previous Period (Spike Detection)
    previous_file = get_previous_processed_file(current_file)
    if previous_file:
        print(f"Comparing with previous data from {previous_file}")
        try:
            previous_df = pd.read_csv(previous_file, parse_dates=["at"])
            previous_overall_sentiment_dist = previous_df['sentiment'].value_counts(normalize=True) * 100
            previous_negative_sentiment_pct = previous_overall_sentiment_dist.get('NEGATIVE', 0.0)

            if previous_negative_sentiment_pct > 0: # Avoid division by zero
                percentage_increase = ((current_negative_sentiment_pct - previous_negative_sentiment_pct) / previous_negative_sentiment_pct) * 100
                if percentage_increase > NEGATIVE_INCREASE_PERCENTAGE:
                    template_data["alerts"].append({
                        'type': 'alert-orange', # Using orange for relative increase alert
                        'message': f"📈 WARNING: Negative sentiment increased by {percentage_increase:.2f}% (from {previous_negative_sentiment_pct:.2f}% to {current_negative_sentiment_pct:.2f}%) compared to previous period."
                    })
                    if not template_data["alerts"]: # Only update subject if no critical alert yet
                        subject = "📈 ALERT: Shein Review Sentiment Trend Change!"
            elif current_negative_sentiment_pct > 0 and previous_negative_sentiment_pct == 0:
                 template_data["alerts"].append({
                    'type': 'alert-orange',
                    'message': f"📈 WARNING: Negative sentiment appeared/increased from 0% to {current_negative_sentiment_pct:.2f}%."
                 })
                 if not template_data["alerts"]:
                     subject = "📈 ALERT: Shein Review Sentiment Trend Change!"

        except Exception as e:
            print(f"Could not load/process previous file {previous_file} for comparison: {e}")
            template_data["alerts"].append({
                'type': 'alert-orange',
                'message': f"Note: Could not compare to previous period due to error: {e}. Please check script logs."
            })
    else:
        print("Not enough historical data for comparison.")
        template_data["alerts"].append({
            'type': 'alert-green', # Using green for informational notes
            'message': "Note: Not enough historical data for trend comparison."
        })

    # Render the HTML email body using Jinja2
    html_body = EMAIL_TEMPLATE.render(template_data)
    print("\n--- HTML Report Generated (Snippet Below) ---\n")
    print(html_body[:500] + "...\n") # Print first 500 chars for preview

    # Send the email
    print("Sending email...")
    send_email(subject, html_body, main_recipients, bcc_recipients=bcc_stakeholders, attachment_path=current_file)

    print("--- Sentiment Notifier Finished ---")

if __name__ == "__main__":
    generate_sentiment_report_and_alerts()