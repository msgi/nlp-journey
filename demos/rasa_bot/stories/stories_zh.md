## simple path with greet
* greet
  - utter_greet
* weather{"address": "上海", "date-time": "明天"}
  - utter_working_on_it
  - action_report_weather
  - utter_report_weather

## simple path
* weather{"address": "上海", "date-time": "明天"}
  - utter_working_on_it
  - action_report_weather
  - utter_report_weather

## address + date-time path with greet
* greet
  - utter_greet
* weather
* weather{"address": "上海"}
  - utter_ask_date-time
* weather{"date-time": "明天"}
  - utter_working_on_it
  - action_report_weather
  - utter_report_weather

## address + date-time path
* weather{"address": "上海"}
  - utter_ask_date-time
* weather{"date-time": "明天"}
  - utter_working_on_it
  - action_report_weather
  - utter_report_weather

## date-time + address path with greet
* greet
  - utter_greet
* weather{"date-time": "明天"}
  - utter_ask_address
* weather{"address": "上海"}
  - utter_working_on_it
  - action_report_weather
  - form{"name": "action_report_weather"}
  - utter_report_weather

## date-time + address path
* weather{"date-time": "明天"}
  - utter_ask_address
* weather{"address": "上海"}
  - utter_working_on_it
  - form{"name": null}
  - action_report_weather
  - utter_report_weather

## say goodbye
* goodbye
  - utter_goodbye