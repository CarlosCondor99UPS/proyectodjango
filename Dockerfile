FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt-get -qq update --fix-missing && \
    apt-get -qq install build-essential && \
    apt-get -qq install libgl1 python3-pip && \
    pip install pycuda && \
    pip install Django && \
    pip install numpy && \
    pip install gunicorn \
    pip install django-cors-headers

WORKDIR /app

# Copia el resto de la aplicación al contenedor (ajustado el directorio de origen)
COPY . .

# Configura las variables de entorno necesarias para Django
ENV DJANGO_SETTINGS_MODULE=filterService.settings

# Expone el puerto en el que se ejecutará la aplicación
EXPOSE 8000

# Comando para ejecutar la aplicación con Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "filterService.wsgi:application"]
