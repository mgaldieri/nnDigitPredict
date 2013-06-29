from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.conf.urls import patterns, include, url
from nnDigitPredict import settings
# Uncomment the next two lines to enable the admin:
from django.contrib import admin
admin.autodiscover()

urlpatterns = patterns('',
    # Examples:
    url(r'^$', 'NeuralPredict.views.home', name='home'),
    url(r'^look$', 'NeuralPredict.views.look', name='look'),
    # url(r'^nnDigitPredict/', include('nnDigitPredict.foo.urls')),

    # Uncomment the admin/doc line below to enable admin documentation:
    url(r'^admin/doc/', include('django.contrib.admindocs.urls')),

    # Uncomment the next line to enable the admin:
    url(r'^admin/', include(admin.site.urls)),

)
urlpatterns += staticfiles_urlpatterns()
