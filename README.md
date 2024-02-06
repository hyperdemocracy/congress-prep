# congress-preproc


# skypilot scraper

Launch skypilot test job,
```shell
sky launch test.yaml --down --idle-minutes-to-autostop 1 --env CONGRESS_NUM=118
```

Launch skypilot scraping job,
```shell
sky launch congress-scraper.yaml --down --idle-minutes-to-autostop 1 --env CONGRESS_NUM=118
```

# copy or sync s3 to local

```shell
aws s3 cp --recursive s3://hyperdemocracy/congress-scraper /local/path/to/congress-scraper
```

```shell
aws s3 sync s3://hyperdemocracy/congress-scraper /local/path/to/congress-scraper
```

