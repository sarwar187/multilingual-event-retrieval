from predpatt import PredPatt

def get_event_windows_simple(text):
    try:
        pp = PredPatt.from_sentence(text)    
        tokens = [token.text for token in pp.tokens]
        windows = []
        predicates = []
        for event in pp.events:
            window_begin = 0
            window_end = 0
            for i, argument in enumerate(event.arguments):        
                #print(argument.coords())
                #print(argument.root)
                #print(argument.phrase())
                #print(argument.position)    
                if i == 0:
                    window_begin = argument.position
                if i == len(event.arguments) - 1:
                    window_end = argument.position 
            window_begin = min(window_begin, event.position)              
            window_end = max(window_end, event.position) + 1
            windows.append(" ".join(tokens[window_begin: window_end]))
            predicates.append(event.root.text)   
        if len(windows) == 0:
            windows.append(text)
            predicates.append(text) 
    except:
        print("Error")
        return [text], [text.split()[0]]            
    return windows, predicates
