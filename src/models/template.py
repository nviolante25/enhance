class EnhancementStage:
    def __init__(self):
        return
    
    @classmethod
    def create(cls, args):
        return cls(**args)
        
    def apply(self):
        raise NotImplementedError
    
    
    
    
