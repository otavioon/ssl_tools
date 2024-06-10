class DomainDataset:
    def __init__(self, dataset, domain):
        self.dataset = dataset
        self.domain = domain
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        if isinstance(item, tuple):
            item = list(item)
        if not isinstance(item, list):
            item = [item]
        item.append(self.domain)
        return tuple(item)
    
    def __len__(self):
        return len(self.dataset)