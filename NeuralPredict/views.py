from django.http import HttpResponse, HttpResponseNotFound
from django.http import Http404
from django.core.context_processors import csrf
from django.shortcuts import render_to_response
from PIL import Image
# from base64 import decodestring
from cStringIO import StringIO
from django.utils import simplejson
from utils import nn
import re, json


def home(request):
    c = {}
    c.update(csrf(request))
    return render_to_response('home.html', c)


def look(request):
    # Handle file upload
    if request.method == 'POST':
        # image = Image.fromstring('RGB',(430,430),decodestring(request.POST['img']))
        image_uri = request.POST['img']
        image_data = re.search(r'base64,(.*)', image_uri).group(1)
        image = Image.open(StringIO(image_data.decode('base64')))
        # image.show()
        # image.save("foo.png")
        number = nn.predict(image)
        response = {
            'predicted': str(number)
        }
        return HttpResponse(json.dumps(response), mimetype='application/json')
