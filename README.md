# Civolux Data Analysis

Dit is een Python [FastAPI](https://fastapi.tiangolo.com/) waar analyses op bouwmaterialen aan kunnen worden opgevraagd. De API heeft drie endpoints voor deze analyses:

1. `/demolitions` voor sloopvoorspellingen
2. `/predict_clusters` voor gebouwcluster-voorspellingen
3. `/predict_twins` voor twinbuilding voorspellingen

Bij het opstarten van de API worden de dataset met gebouwen, en de dataset met materialen ingeladen. Ook wordt direct op alle gebouwen een Random Forest Classifier uitgevoerd die gebruikt wordt voor de sloopvoorspellingen. Hierdoor duurt het enkele minuten voordat de API volledig opgestart is.

Voor de gebouwcluster-voorspellingen wordt een KMeans Clustering model gebruikt. De twinbuilding voorspellingen gebruiken geen machine learning model, maar matchen gebouwen simpelweg op enkele criteria.

### Overige endpoints

Daarnaast zijn er nog een paar endpoints ter ondersteuning:

- `/materials` om de totale hoeveelheden voor alle materialen op te halen. Deze worden berekend aan de hand van de gemiddelde hoeveelheden per type materiaal en het aantal gebouwen.
- `/building_types` om alle pandtypes op te halen.
- `/addresses` om adressen op te halen op basis van een zoekopdracht.
- `/health` om de gezondheid van de API te kunnen controleren. De response geeft ook aan of alle datasets geladen zijn.

## Opstarten

### Maak een virtual environment (optioneel)

Om conflicten met andere globale Python packages te vermijden is het aangeraden om eerst een virtual environment te gebruiken:

`python -m venv .venv`

Activeer de virtual environment:

`.venv\Scripts\activate`

### Installeer Python packages

`pip install -r requirements.txt`

### Start de API

In de Dockerfile wordt `gunicorn` (Green Unicorn) gebruikt voor de HTTP-server. Dit is een betere versie van `uvicorn` voor productie, omdat deze meerdere workers ondersteunt. Dit is echter niet vereist voor development, aangezien dit meerdere instances van de API draait.

Aanbevolen voor development:

`uvicorn main:app`

Door aan het bovenstaande commando een `--reload` flag toe te voegen wordt de API na iedere aanpassingen opnieuw opgestart. Dit kan handig zijn tijdens het testen van aanpassingen, behalve als de API lang duurt om op te starten, zoals momenteel het geval is. Het commando wordt dan:

`uvicorn main:app --reload`

Om meerdere instances te kunnen draaien (bijvoorbeeld voor productie) kan het onderstaande commando worden gebruikt:

`gunicorn main:app --workers <aantal workers> --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000`