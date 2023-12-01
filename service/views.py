from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import json
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time
from django.http import HttpResponse

import base64

@csrf_exempt
@require_POST
def cifrar_texto_AES(request):
    response = HttpResponse()
    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response["Access-Control-Allow-Headers"] = "Origin, Content-Type, Accept"
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        # Obtener texto plano y clave del JSON
        texto_plano = data.get('texto', '')
        clave = data.get('clave', '')

        # Verificar que la clave sea de 16 caracteres
        if len(clave) != 16:
            return JsonResponse({'error': 'La clave debe tener exactamente 16 caracteres.'}, status=400)
        
        cuda.init()
        device = cuda.Device(0) 
        context = device.make_context()

        mod = """
#define N  64

// Definir funciones para evitar errores de compilación
__device__ void addRoundKey(unsigned int* vector_claves, unsigned int* vector_estado, int n);
__device__ void subBytes(unsigned int* vector_estado);
__device__ void shiftRows(unsigned int* vector_estado);
__device__ void mixColumns(unsigned int* vector_estado);

__global__ void aes_funciones_codifica_kernel(unsigned int* vector_estado, unsigned int* vector_claves)
{
    int l = 0, n;
    n = 0;
    addRoundKey(vector_claves, vector_estado, n);
    for (l = 0; l <= 8; l++)
    {
        subBytes(vector_estado);
        shiftRows(vector_estado);
        mixColumns(vector_estado);
        n = n + 1;
        addRoundKey(vector_claves, vector_estado, n);
    }
    subBytes(vector_estado);
    shiftRows(vector_estado);
    n = n + 1;
    addRoundKey(vector_claves, vector_estado, n);
}

__device__ void addRoundKey(unsigned int* vector_claves, unsigned int* vector_estado, int n){
    int i;
    int tID = threadIdx.x;
    for (i = 0; i < 4; i++)
    {
        if (tID < 16)
        {
           vector_estado[tID+(16*i)]=vector_estado[tID+(16*i)]^vector_claves[(16*n)+tID];               
    
        }
        __syncthreads();
    }
}


__device__ void subBytes(unsigned int* vector_estado)
{
    int tID = threadIdx.x;
    int sbox[256] = {
        
        //0 1 2 3 4 5 6 7 8 9 A B C D E F
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76, //0
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0, //1
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15, //2
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75, //3
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84, //4
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf, //5
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8, //6
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2, //7
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73, //8
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb, //9
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79, //A
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08, //B
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a, //C
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e, //D
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf, //E
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16};//F

    if (tID < N){
        vector_estado[tID] = sbox[vector_estado[tID]];
    }
}

__device__ void shiftRows(unsigned int* vector_estado)
{
    int tID = threadIdx.x;
    int i;
    unsigned int estado_auxiliar[N];
    for (i = 0; i < 4; i++)
    {
        // Primera fila
        estado_auxiliar[(16*i) + 0] = vector_estado[(16*i) + 0];
        estado_auxiliar[(16*i) + 1] = vector_estado[(16*i) + 1];
        estado_auxiliar[(16*i) + 2] = vector_estado[(16*i) + 2];
        estado_auxiliar[(16*i) + 3] = vector_estado[(16*i) + 3];
        // Segunda fila
        estado_auxiliar[(16*i) + 4] = vector_estado[(16*i) + 5];
        estado_auxiliar[(16*i) + 5] = vector_estado[(16*i) + 6];
        estado_auxiliar[(16*i) + 6] = vector_estado[(16*i) + 7];
        estado_auxiliar[(16*i) + 7] = vector_estado[(16*i) + 4];
        // Tercera fila
        estado_auxiliar[(16*i) + 8] = vector_estado[(16*i) + 10];
        estado_auxiliar[(16*i) + 9] = vector_estado[(16*i) + 11];
        estado_auxiliar[(16*i) + 10] = vector_estado[(16*i) + 8];
        estado_auxiliar[(16*i) + 11] = vector_estado[(16*i) + 9];
        // Cuarta fila
        estado_auxiliar[(16*i) + 12] = vector_estado[(16*i) + 15];
        estado_auxiliar[(16*i) + 13] = vector_estado[(16*i) + 12];
        estado_auxiliar[(16*i) + 14] = vector_estado[(16*i) + 13];
        estado_auxiliar[(16*i) + 15] = vector_estado[(16*i) + 14];
    }
    if (tID < N)
    {
  
         vector_estado[tID] = estado_auxiliar[tID];

    }
}

__device__ void mixColumns(unsigned int* vector_estado)
{
    int i;
    int tID = threadIdx.x;
    unsigned char Tmep, Tme, time;

    #define xtime(x) ((x << 1) ^ (((x >> 7) & 1) * 0x1b))

    for (i = 0; i < 4; i++)
    {
        if (tID < 4)
        {
            time = vector_estado[(16*i) + tID];
            Tmep = vector_estado[(16*i) + tID] ^ vector_estado[(16*i) + 4 + tID] ^ vector_estado[(16*i) + 8 + tID] ^ vector_estado[(16*i) + 12 + tID];
            Tme = vector_estado[(16*i) + tID] ^ vector_estado[(16*i) + 4 + tID];
            Tme = xtime(Tme);
            vector_estado[(16*i) + 0 + tID] ^= Tme ^ Tmep;
            Tme = vector_estado[(16*i) + 4 + tID] ^ vector_estado[(16*i) + 8 + tID];
            Tme = xtime(Tme);
            vector_estado[(16*i) + 4 + tID] ^= Tme ^ Tmep;
            Tme = vector_estado[(16*i) + 8 + tID] ^ vector_estado[(16*i) + 12 + tID];
            Tme = xtime(Tme);
            vector_estado[(16*i) + 8 + tID] ^= Tme ^ Tmep;
            Tme = vector_estado[(16*i) + 12 + tID] ^ time;
            Tme = xtime(Tme);
            vector_estado[(16*i) + 12 + tID] ^= Tme ^ Tmep;
        }
    }
}
"""
        # Convertir texto plano y clave a bytes
        vector_estado = np.array([ord(char) for char in texto_plano], dtype=np.uint32)

        N = vector_estado.shape[0]

        clave = np.array([ord(char) for char in clave], dtype=np.uint32)

        print(vector_estado)
        print(clave)

        mod = mod.replace("#define N 64", f"#define N {N}")
        # Funciones CUDA
        mod = SourceModule(mod)


        # Obtener funciones del módulo
        aes_funciones_codifica_kernel = mod.get_function("aes_funciones_codifica_kernel")

        # Reservar memoria en la GPU
        dev_vector_estado = cuda.mem_alloc(N * np.uint32().itemsize)
        dev_vector_claves = cuda.mem_alloc(176 * np.uint32().itemsize)


        block = (128, 1, 1)  
        grid = ((N + block[0] - 1) // block[0], 1)

        # Copiar datos a la GPU
        cuda.memcpy_htod(dev_vector_estado, vector_estado)
        cuda.memcpy_htod(dev_vector_claves, clave)
        
        # Medir el tiempo de ejecución
        start_time = time.time()

        # Invocar al kernel
        aes_funciones_codifica_kernel(dev_vector_estado, dev_vector_claves, block=block, grid=grid)

      

        # Obtener resultados
        vector_resultado_host = np.empty(N, dtype=np.uint32)
        cuda.memcpy_dtoh(vector_resultado_host, dev_vector_estado)
        # Calcular el tiempo de ejecución
        elapsed_time = time.time() - start_time

        print("Tiempo de ejecución: ",elapsed_time)

        print("Resultado después de cifrar:")
        print(vector_resultado_host)

        context.pop()

        # Devolver el resultado en la respuesta JSON
        return JsonResponse({'resultado': vector_resultado_host.tolist()})
    
    else:
        return JsonResponse({'error': 'Solo se permite el método POST'})
    
@csrf_exempt
@require_POST
def decifrar_texto_AES(request):
    response = HttpResponse()
    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response["Access-Control-Allow-Headers"] = "Origin, Content-Type, Accept"
    if request.method == 'POST':
        cuda.init()
        device = cuda.Device(0) 
        context = device.make_context()

        data = json.loads(request.body.decode('utf-8'))
        # Obtener texto plano y clave del JSON
        texto_plano = data.get('texto', '')
        clave = data.get('clave', '')

        valores_str = texto_plano.split(",")
        
        mod = """
#define N  64


__device__ void addRoundkey(unsigned int* vector_claves, unsigned int* vector_estado, int n);
__device__ void inv_subBytes(unsigned int* vector_estado);
__device__ void inv_shiftRows(unsigned int* vector_estado);
__device__ void inv_mixColumns(unsigned int* vector_estado);

__global__ void aes_funciones_descodifica_kernel (unsigned int*vector_estado, unsigned
int*vector_claves) {
    int n=10;
    int l;
    {
    addRoundkey(vector_claves, vector_estado, n);
    for(l=0;l<=8;l++){
        inv_shiftRows(vector_estado);
        inv_subBytes(vector_estado);
        n=n-1;
        addRoundkey(vector_claves, vector_estado, n);
        inv_mixColumns(vector_estado);

    }
    
        inv_shiftRows(vector_estado);
        inv_subBytes(vector_estado);
        n=n-1;
        addRoundkey(vector_claves, vector_estado, n);
        n=10;
    }

}

__device__ void addRoundkey(unsigned int* vector_claves, unsigned int* vector_estado, int n){
    int i;
    int tID = threadIdx.x;
    for (i = 0; i < 4; i++)
    {
        if (tID < 16)
        {
           vector_estado[tID+(16*i)]=vector_estado[tID+(16*i)]^vector_claves[(16*n)+tID];   
        }

        __syncthreads();
    
    }

    
}


__device__ void inv_shiftRows(unsigned int* vector_estado){
    int tID=threadIdx.x;
    int i;
    unsigned int estado_auxiliar[N];
    for(i=0;i<4;i++){
    
        //Primera fila
        estado_auxiliar[(16*i)+ 0]=vector_estado[(16*i)+0];
        estado_auxiliar[(16*i)+1]=vector_estado[(16*i)+1];
        estado_auxiliar[(16*i)+2]=vector_estado[(16*i)+2];
        estado_auxiliar[(16*i)+3]=vector_estado[(16*i)+3];
        //Segunda fila
        estado_auxiliar[(16*i)+4]=vector_estado[(16*i)+7];
        estado_auxiliar[(16*i)+5]=vector_estado[(16*i)+4];
        estado_auxiliar[(16*i)+6]=vector_estado[(16*i)+5];
        estado_auxiliar[(16*i)+7]=vector_estado[(16*i)+6];
        //Tercera fila
        estado_auxiliar[(16*i)+8]=vector_estado[(16*i)+10];
        estado_auxiliar[(16*i)+9]=vector_estado[(16*i)+11];
        estado_auxiliar[(16*i)+10]=vector_estado[(16*i)+8];
        estado_auxiliar[(16*i)+11]=vector_estado[(16*i)+9];
        //Cuarta fila
        estado_auxiliar[(16*i)+12]=vector_estado[(16*i)+13];
        estado_auxiliar[(16*i)+13]=vector_estado[(16*i)+14];
        estado_auxiliar[(16*i)+14]=vector_estado[(16*i)+15];
        estado_auxiliar[(16*i)+15]=vector_estado[(16*i)+12];
    }

        if (tID < N){
        
            vector_estado[tID]=estado_auxiliar[tID];

        }

 }       

 __device__ void inv_subBytes(unsigned int* vector_estado){
    int tID=threadIdx.x;
    int isbox[256] = {
        //0 1 2 3 4 5 6 7 8 9 A B C D E F
        0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb, //0
        0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb, //1
        0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e, //2
        0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25, //3
        0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92, //4
        0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84, //5
        0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06, //6
        0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b, //7
        0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73, //8
        0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e, //9
        0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b, //A
        0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4, //B
        0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f, //C
        0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef, //D
        0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61, //E
        0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
        
    };

        if (tID < N){
        
            vector_estado[tID]=isbox[vector_estado[tID]];

        }

 }

 __device__ void inv_mixColumns (unsigned int* vector_estado){
 
    int tID=threadIdx.x;
    int i;
    #define xtime(x) ((x<<1) ^ (((x>>7) & 1) * 0x1b))
    int auxiliar[N], num_a,num_b,num_c,num_d;
    #define Multiply(x,y) (((y & 1) * x) ^ ((y>>1 & 1) * xtime(x)) ^ ((y>>2 & 1) * xtime(xtime(x))) ^ ((y>>3 & 1) * xtime(xtime(xtime(x)))) ^ ((y>>4 & 1) * xtime(xtime(xtime(xtime(x))))))
    for(i=0;i<4;i++){
        if(tID<4){
            num_a = vector_estado[(16*i)+tID];
            num_b = vector_estado[(16*i+4)+tID];
            num_c = vector_estado[(16*i+8)+tID];
            num_d = vector_estado[(16*i+12)+tID];
            vector_estado[(16*i)+tID] = Multiply(num_a, 0x0e) ^ Multiply(num_b, 0x0b) ^
            Multiply(num_c, 0x0d) ^ Multiply(num_d, 0x09);
            vector_estado[(16*i+4)+tID] = Multiply(num_a, 0x09) ^ Multiply(num_b, 0x0e) ^
            Multiply(num_c, 0x0b) ^ Multiply(num_d, 0x0d);
            vector_estado[(16*i+8)+tID] = Multiply(num_a, 0x0d) ^ Multiply(num_b, 0x09) ^
            Multiply(num_c, 0x0e) ^ Multiply(num_d, 0x0b);
            vector_estado[(16*i+12)+tID] = Multiply(num_a, 0x0b) ^ Multiply(num_b, 0x0d) ^
            Multiply(num_c, 0x09) ^ Multiply(num_d, 0x0e);
        } 
    }

    __syncthreads();

    if (tID < N){
    
        auxiliar[tID]=vector_estado[tID]/0x100;
        vector_estado[tID]=vector_estado[tID]-auxiliar[tID]*0x100;
    
    }

 }

"""
        # Convertir cadenas hexadecimales a números enteros
        vector_estado = np.array(valores_str, dtype=np.uint32)
        N = vector_estado.shape[0]

        clave = np.array([ord(char) for char in clave], dtype=np.uint32)


        print(vector_estado)
        print(clave)

        mod = mod.replace("#define N 64", f"#define N {N}")
        # Funciones CUDA
        mod = SourceModule(mod)

        # Obtener funciones del módulo
        aes_funciones_codifica_kernel = mod.get_function("aes_funciones_descodifica_kernel")

        dev_vector_estado = cuda.mem_alloc(N * np.uint32().itemsize)
        dev_vector_claves = cuda.mem_alloc(176 * np.uint32().itemsize)

        # Configurar las dimensiones del bloque y la cuadrícula
        block = (128, 1, 1)  # Ajusta el tamaño del bloque según tu GPU
        grid = ((N + block[0] - 1) // block[0], 1)

        # Copiar datos a la GPU
        cuda.memcpy_htod(dev_vector_estado, vector_estado)
        cuda.memcpy_htod(dev_vector_claves, clave)

        # Medir el tiempo de ejecución
        start_time = time.time()

        # Invocar al kernel
        aes_funciones_codifica_kernel(dev_vector_estado, dev_vector_claves, block=block, grid=grid)


        # Obtener resultados
        vector_resultado_host = np.empty(N, dtype=np.uint32)
        cuda.memcpy_dtoh(vector_resultado_host, dev_vector_estado)
        # Calcular el tiempo de ejecución
        elapsed_time = time.time() - start_time

        print("Tiempo de ejecucion: ", elapsed_time)

        print("Resultado después de cifrar:")
        print(vector_resultado_host)

        # Liberar memoria en la GPU
        dev_vector_estado.free()
        dev_vector_claves.free()

        # Imprimir el array en formato hexadecimal
        print("Vector en formato hexadecimal:", [hex(valor) for valor in vector_resultado_host])

        # Convertir los valores a cadena de texto y unirlos
        cadena_resultado = ''.join([chr(valor) for valor in vector_resultado_host])


        context.pop()
        # Imprimir la cadena de texto resultante
        print("Cadena de texto generada:", cadena_resultado)


        # Devolver el resultado en la respuesta JSON
        return JsonResponse({'resultado': str(cadena_resultado)})


    else:
        return JsonResponse({'error': 'Solo se permite el método POST'})
    

@csrf_exempt
@require_POST
def cifrar_texto_RSA(request):
    response = HttpResponse()
    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response["Access-Control-Allow-Headers"] = "Origin, Content-Type, Accept"

    if request.method == 'POST':
        try:
            # Obtener el texto y claves RSA desde el JSON
            cuda.init()
            device = cuda.Device(0)
            context = device.make_context()
            data = json.loads(request.body.decode('utf-8'))
            texto_plano = data.get('texto', '')
            private_key = data.get('d', '')
            public_key = data.get('N', '')

            # Convertir el mensaje a una lista de enteros
            message_int = [ord(char) for char in texto_plano]

            # Definir el kernel
            mod = SourceModule("""
                __device__ unsigned long long int mod_pow(unsigned long long int base, unsigned long long int exponent, unsigned long long int modulus) {
                    unsigned long long int result = 1;
                    base = base % modulus;

                    while (exponent > 0) {
                        if (exponent % 2 == 1) {
                            result = (result * base) % modulus;
                        }

                        exponent = exponent >> 1;
                        base = (base * base) % modulus;
                    }

                    return result;
                }

                __global__ void rsa_encrypt_kernel(int *message, int e, int N, int *result, int length) {
                    int i = threadIdx.x;
                    if (i < length) {
                        result[i] = mod_pow(message[i], e, N);
                    }
                }
            """)

            # Obtener la función del kernel
            rsa_encrypt_kernel = mod.get_function("rsa_encrypt_kernel")

            # Parámetros e y N
            e = np.int64(public_key)
            N = np.int64(private_key)

            # Crear arrays de GPU y copiar datos desde el host
            message_host = np.array(message_int, dtype=np.int32)
            message_gpu = cuda.to_device(message_host)
            result_gpu = cuda.mem_alloc(message_host.nbytes)

            # Llamar al kernel
            block_size = len(message_int)
            # Medir el tiempo de ejecución
            start_time = time.time()
            rsa_encrypt_kernel(message_gpu, np.int32(e), np.int32(N), result_gpu, np.int32(block_size), block=(block_size, 1, 1), grid=(1, 1))
            # Calcular el tiempo de ejecución
            elapsed_time = time.time() - start_time

            print("Tiempo de ejecución: ",elapsed_time)
            # Copiar el resultado de vuelta al host
            result_host = np.empty_like(message_host)
            cuda.memcpy_dtoh(result_host, result_gpu)

            # Aquí puedes imprimir o manejar de alguna manera el resultado del cifrado RSA
            print("Texto cifrado con RSA:", result_host)
            context.pop()
            # Devolver la respuesta JSON o realizar otras acciones según sea necesario
            return JsonResponse({'resultado': result_host.tolist()})

        except Exception as e:
            # Manejar cualquier error que pueda ocurrir durante el proceso
            return JsonResponse({'error': str(e)})

    else:
        return JsonResponse({'error': 'Solo se permite el método POST'})


@csrf_exempt
@require_POST
def descifrar_texto_RSA(request):
    response = HttpResponse()
    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response["Access-Control-Allow-Headers"] = "Origin, Content-Type, Accept"

    if request.method == 'POST':
        try:
            # Obtener el texto cifrado y claves RSA desde el JSON
            cuda.init()
            device = cuda.Device(0)
            context = device.make_context()
            data = json.loads(request.body.decode('utf-8'))
            texto_cifrado = data.get('resultado', [])  # Cambiado a 'resultado' para el texto cifrado
            private_key = data.get('d', '')  # Cambiado a 'd' para la clave privada
            public_key = data.get('N', '')

            # Definir el kernel
            mod = SourceModule("""
                __device__ unsigned long long int mod_pow(unsigned long long int base, unsigned long long int exponent, unsigned long long int modulus) {
                    unsigned long long int result = 1;
                    base = base % modulus;

                    while (exponent > 0) {
                        if (exponent % 2 == 1) {
                            result = (result * base) % modulus;
                        }

                        exponent = exponent >> 1;
                        base = (base * base) % modulus;
                    }

                    return result;
                }

                __global__ void rsa_decrypt_kernel(int *ciphertext, int d, int N, int *result, int length) {
                    int i = threadIdx.x;
                    if (i < length) {
                        result[i] = mod_pow(ciphertext[i], d, N);
                    }
                }
            """)

            # Obtener la función del kernel
            rsa_decrypt_kernel = mod.get_function("rsa_decrypt_kernel")

            # Parámetros d y N
            d = np.int64(private_key)
            N = np.int64(public_key)

            # Calcular el tamaño del bloque (puedes ajustar esto según sea necesario)
            block_size = 128

            # Calcular la cantidad de bloques necesarios para procesar todo el texto cifrado
            num_blocks = (len(texto_cifrado) + block_size - 1) // block_size

            # Inicializar el resultado como una lista
            result_host = []
            # Medir el tiempo de ejecución
            start_time = time.time()
            # Iterar sobre cada bloque
            for block_start in range(0, len(texto_cifrado), block_size):
                block_end = min(block_start + block_size, len(texto_cifrado))

                # Obtener el bloque actual
                current_block = texto_cifrado[block_start:block_end]

                # Crear arrays de GPU y copiar datos desde el host (para el bloque actual)
                current_block_host = np.array(current_block, dtype=np.int32)
                current_block_gpu = cuda.to_device(current_block_host)
                result_block_gpu = cuda.mem_alloc(current_block_host.nbytes)

                # Llamar al kernel (para el bloque actual)
                rsa_decrypt_kernel(current_block_gpu, np.int32(d), np.int32(N), result_block_gpu, np.int32(len(current_block)), block=(len(current_block), 1, 1), grid=(1, 1))

                # Copiar el resultado de vuelta al host (para el bloque actual)
                result_block_host = np.empty_like(current_block_host)
                cuda.memcpy_dtoh(result_block_host, result_block_gpu)

                # Extender el resultado con el resultado del bloque actual
                result_host.extend(result_block_host)
            # Calcular el tiempo de ejecución
            elapsed_time = time.time() - start_time

            print("Tiempo de ejecución: ",elapsed_time)
            # Convertir la lista de enteros a caracteres y concatenarlos para obtener el texto descifrado
            texto_descifrado = ''.join([chr(char) for char in result_host])

            # Aquí puedes imprimir o manejar de alguna manera el resultado del descifrado RSA
            print("Texto descifrado con RSA:", texto_descifrado)
            context.pop()
            # Devolver la respuesta JSON o realizar otras acciones según sea necesario
            return JsonResponse({'texto_descifrado': texto_descifrado})

        except Exception as e:
            # Manejar cualquier error que pueda ocurrir durante el proceso
            return JsonResponse({'error': str(e)})

    else:
        return JsonResponse({'error': 'Solo se permite el método POST'})
