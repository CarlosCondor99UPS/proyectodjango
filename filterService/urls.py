from django.urls import path
from service.views import cifrar_texto_AES
from service.views import decifrar_texto_AES
from service.views import descifrar_texto_RSA
from service.views import cifrar_texto_RSA

urlpatterns = [
    # Otras rutas pueden ir aquí
    path('cifrar_texto_AES/', cifrar_texto_AES, name='cifrar_texto_AES'),
    path('decifrar_texto_AES/', decifrar_texto_AES, name='decifrar_texto_AES'),
    path('cifrar_texto_RSA/', cifrar_texto_RSA, name='cifrar_texto_RSA'),
    path('decifrar_texto_RSA/', descifrar_texto_RSA, name='decifrar_texto_RSA'),

]