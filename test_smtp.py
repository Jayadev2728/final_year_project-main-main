import smtplib
try:
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as server:
        print("Connected successfully!")
        server.starttls()
        print("TLS started successfully!")
except Exception as e:
    print(f"Failed: {e}")