# Render deployment notes

This project is ready for a basic Render web service deploy.

## Important limitation
This app currently uses a local SQLite database file:
`database/expense_tracker.db`

On Render free web services, local files are ephemeral. That means user data can be lost after:
- redeploys
- restarts
- idle spin-downs

So this setup is fine for a demo, portfolio, or testing, but not for real persistent user data.

## Deploy steps
1. Push this folder to a GitHub repo.
2. In Render, create a new Web Service from that repo.
3. Render can detect `render.yaml`, or you can set:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
4. Add any extra environment variables if needed.
5. Deploy.

## Secret key
The app now reads `SECRET_KEY` from environment variables when provided.
