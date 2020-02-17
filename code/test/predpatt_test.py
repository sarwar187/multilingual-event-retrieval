from predpatt import PredPatt

pp = PredPatt.from_sentence('At the Pentagon briefing today, General Stanley McChrystal said that it looked a lot like terrorism.')
#print(pp.pprint())
# print(" ".join([token.text for token in pp.tokens]))
# print(pp.events)
# print(pp.event_dict)
# print(pp.events)



for event in pp.events:
    print(event)
    for argument in event.arguments:
        print(argument)