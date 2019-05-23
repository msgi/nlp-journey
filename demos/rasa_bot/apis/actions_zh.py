from rasa_core_sdk.events import SlotSet
from rasa_core_sdk.forms import FormAction
from requests import ConnectionError, HTTPError, TooManyRedirects, Timeout
import requests

import json

KEY = '4r9bergjetiv1tsd'
UID = "U785B76FC9"

LOCATION = 'beijing'
API = 'https://api.seniverse.com/v3/weather/now.json'
UNIT = 'c'
LANGUAGE = 'zh-Hans'


def get_weather_by_day(location):
    try:
        result = requests.get(API, params={
            'key': KEY,
            'location': location,
            'language': LANGUAGE,
            'unit': UNIT
        }, timeout=2)
        result = result.text
    except (ConnectionError, HTTPError, TooManyRedirects, Timeout) as e:
        result = "{}".format(e)

    result = json.loads(result)
    normal_result = {
        "location": result["results"][0]["location"]['path'],
        "result": result["results"][0]["now"]['text']
    }
    print(normal_result)
    return normal_result


class ReportWeatherAction(FormAction):
    RANDOMIZE = True

    @staticmethod
    def required_slots(tracker):
        return ['address', 'date-time']

    def name(self):
        return "action_report_weather"

    def submit(self, dispatcher, tracker, domain):
        address = tracker.get_slot('address')
        date_time = tracker.get_slot('date-time')
        print(date_time)
        weather_data = get_weather_by_day(address)
        return [SlotSet("matches", "{}".format(weather_data))]