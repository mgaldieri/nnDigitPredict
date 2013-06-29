from django.http import HttpResponse, HttpResponseNotFound
from django.http import Http404
from django.core.context_processors import csrf
from django.shortcuts import render_to_response
from PIL import Image
from base64 import decodestring
from django.utils import simplejson

def home(request):
    c = {}
    c.update(csrf(request))
    return render_to_response('home.html', c)
    
def look(request):
    # Handle file upload
    if request.method == 'POST':
        image = Image.fromstring('RGB',(430,430),decodestring(request.POST['img']))
        image.save("foo.png")
        some_data = {
           'some_var_1': 'foo',
           'some_var_2': 'bar',
        }

        data = simplejson.dumps(data)
        return HttpResponse(data, mimetype='application/json')
