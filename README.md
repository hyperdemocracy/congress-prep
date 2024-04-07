# congress-prep


Tools to bulk download and process legislative XML.


# Playbook

## Install [congress](https://github.com/unitedstates/congress.git)

```bash
git clone https://github.com/unitedstates/congress.git
pip install -e congress
```

## Update XML files on download machine

```bash
./bulk_download/download-and-sync-with-log.sh
```

## Sync to local machine

```bash
aws s3 sync s3://hyperdemocracy/congress-scraper congress-scraper
```


## Upsert into postgres

```bash
python congress_prep/01_populate_postgres.py
```


