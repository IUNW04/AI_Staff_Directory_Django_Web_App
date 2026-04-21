import os
from pathlib import Path
import dj_database_url
import cloudinary
import cloudinary.uploader
import cloudinary.api

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get('SECRET_KEY', 'django-insecure-j2x8^z9!4ni5$#@kq7u&!v3nwjp8zk4v9y%m6l*x+p(q)w3e&h')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.environ.get('DEBUG', 'True') == 'True'

ALLOWED_HOSTS = ['*']

# Application definition
INSTALLED_APPS = [
	'django.contrib.admin',
	'django.contrib.auth',
	'django.contrib.contenttypes',
	'django.contrib.sessions',
	'django.contrib.messages',
	'django.contrib.staticfiles',
	'WhosWhoApp',
	'crispy_forms',
	'cloudinary_storage',
	'cloudinary',
]

MIDDLEWARE = [
	'django.middleware.security.SecurityMiddleware',
	'whitenoise.middleware.WhiteNoiseMiddleware',
	'django.contrib.sessions.middleware.SessionMiddleware',
	'django.middleware.common.CommonMiddleware',
	'django.middleware.csrf.CsrfViewMiddleware',
	'django.contrib.auth.middleware.AuthenticationMiddleware',
	'django.contrib.messages.middleware.MessageMiddleware',
	'django.middleware.clickjacking.XFrameOptionsMiddleware',
	'WhosWhoApp.middleware.AdminAccessMiddleware',
]

ROOT_URLCONF = 'FYP_whos_who.urls'

TEMPLATES = [
	{
		'BACKEND': 'django.template.backends.django.DjangoTemplates',
		'DIRS': [],
		'APP_DIRS': True,
		'OPTIONS': {
			'context_processors': [
				'django.template.context_processors.debug',
				'django.template.context_processors.request',
				'django.contrib.auth.context_processors.auth',
				'django.contrib.messages.context_processors.messages',
			],
		},
	},
]

WSGI_APPLICATION = 'FYP_whos_who.wsgi.application'

# Database configuration
DATABASES = {
	'default': {
		'ENGINE': 'django.db.backends.sqlite3',
		'NAME': BASE_DIR / 'db.sqlite3',
	}
}

if os.environ.get('DATABASE_URL'):
	DATABASES['default'] = dj_database_url.config(
		conn_max_age=0,
		conn_health_checks=True,
		ssl_require=True,
		engine='django.db.backends.postgresql',
	)
	DATABASES['default'].update({
		'OPTIONS': {
			'sslmode': 'require',
		},
		'ATOMIC_REQUESTS': True,
		'AUTOCOMMIT': True,
	})



# Password validation
AUTH_PASSWORD_VALIDATORS = [
	{
		'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
	},
	{
		'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
	},
	{
		'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
	},
	{
		'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
	},
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files configuration
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'WhosWhoApp/static')]

# Use different storage backends based on DEBUG mode
if DEBUG:
	STATICFILES_STORAGE = 'django.contrib.staticfiles.storage.StaticFilesStorage'
else:
	STATICFILES_STORAGE = 'whitenoise.storage.CompressedStaticFilesStorage'

# Cloudinary configuration
CLOUDINARY_STORAGE = {
    'CLOUD_NAME': os.environ.get('CLOUDINARY_CLOUD_NAME'),
    'API_KEY': os.environ.get('CLOUDINARY_API_KEY'),
    'API_SECRET': os.environ.get('CLOUDINARY_API_SECRET')
}

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# Update media storage settings
if not DEBUG:
    DEFAULT_FILE_STORAGE = 'cloudinary_storage.storage.MediaCloudinaryStorage'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Login URL
LOGIN_URL = 'login'

# Hugging Face settings
HUGGINGFACE_API_TOKEN = os.environ.get('HUGGINGFACE_API_TOKEN', '')
HUGGINGFACE_MODEL_NAME = os.environ.get('HUGGINGFACE_MODEL_NAME', 'deepseek-ai/DeepSeek-V3')