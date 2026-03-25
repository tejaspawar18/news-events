from cassandra.cqlengine.usertype import UserType
from cassandra.cqlengine import columns
class EventGeoLocationUDT:
    def __init__(self, name, id):
        self.name = name
        self.id = id

class EventPartyUDT:
    def __init__(self, party, party_id):
        self.party = party
        self.party_id = party_id

class EventPartySentimentUDT:
    def __init__(self, sentiment, reason):
        self.sentiment = sentiment
        self.reason = reason


class NamedId(UserType):
    name = columns.Text()
    id = columns.Text()

class Party(UserType):
    party = columns.Text()
    party_id = columns.Text()

class Inclination(UserType):
    sentiment = columns.Text()
    reason = columns.Text()
