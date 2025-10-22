from app import app

# Gunicorn and other WSGI servers will import `application` from this module.
application = app
