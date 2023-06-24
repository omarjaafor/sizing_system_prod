# coding: utf-8

"""
 Camaieu - Markdown Optimisation
 App: server
"""

from flask import Flask


class CamaieuApplicationServer(Flask):
    def process_response(self, response):
        super(CamaieuApplicationServer, self).process_response(response)
        response.headers["Server"] = "Camaieu App Server"
        return response
