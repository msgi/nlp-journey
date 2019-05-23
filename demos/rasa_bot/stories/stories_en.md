
## happy path              
* greet              
  - utter_greet
* mood_great               
  - utter_happy
* mood_affirm
  - utter_happy
* mood_affirm
  - utter_goodbye
  
## sad path 1             
* greet
  - utter_greet             
* mood_unhappy
  - utter_ask_picture
* inform{"animal":"dog"}  
  - action_retrieve_image
  - utter_did_that_help
* mood_affirm
  - utter_happy

## sad path 2
* greet
  - utter_greet
* mood_unhappy
  - utter_ask_picture
* inform{"group":"cat"}
  - action_retrieve_image
  - utter_did_that_help
* mood_deny
  - utter_goodbye
  
## sad path 3
* greet
  - utter_greet
* mood_unhappy{"group":"puppy"}
  - action_retrieve_image
  - utter_did_that_help
* mood_affirm
  - utter_happy
  
## strange user
* mood_affirm
  - utter_happy
* mood_affirm
  - utter_unclear

## say goodbye
* goodbye
  - utter_goodbye

## fallback
- utter_unclear

## Generated Story 8602554595375725553
* greet
* mood_unhappy
    - utter_ask_picture
* inform{"group": "shibes"}
    - slot{"group": "shibes"}
    - action_retrieve_image
    - utter_did_that_help

## Generated Story -5796817115678948843
* greet
* mood_unhappy
    - utter_ask_picture
* inform{"group": "shibes"}
    - slot{"group": "shibes"}
    - action_retrieve_image
    - utter_did_that_help
* greet
* goodbye

## Generated Story 5123535013477899764
* greet
* mood_unhappy
    - utter_ask_picture
* inform{"group": "shibes"}
    - slot{"group": "shibes"}
* goodbye

## Generated Story 283891429498916947
* greet
* mood_unhappy
    - utter_ask_picture
* inform{"group": "cats"}
    - slot{"group": "cats"}
* goodbye

## Generated Story -8825362599989410207
* greet
* mood_unhappy
    - utter_ask_picture

