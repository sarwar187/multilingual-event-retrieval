class ACE: 
    def __init__(self, config):
        self.annotator_name = config["annotator"]["name"]
        self.token = config["annotator"]["token"]
        self.threshold = config["annotator"]["threshold"]
        self.query_description = ""

    def create_queries(self, config):
        """[summary]
        
        Arguments:
            config {[configuration file]} -- [query language]
        """
        