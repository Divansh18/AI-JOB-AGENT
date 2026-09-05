from jobsearch.sources.manual import build_posting


def test_manual_lever_ingest_extracts_header_company_title_and_location():
    html = """
    <html>
      <head><title>Hevo Data - SDE I</title></head>
      <body>
        SDE I Bangalore, India Engineering - Backend / Full time / On-site
        apply for this job
        What you'll own as SDE I at Hevo: Build SaaS connectors.
      </body>
    </html>
    """

    posting = build_posting("https://jobs.lever.co/hevodata/abc123", html)

    assert posting.company_name == "Hevo Data"
    assert posting.title == "SDE I"
    assert posting.location_raw == "Bangalore, India"


def test_manual_lever_ingest_preserves_overrides():
    html = """
    <html>
      <head><title>Hevo Data - SDE I</title></head>
      <body>SDE I Bangalore, India Engineering - Backend / Full time / On-site</body>
    </html>
    """

    posting = build_posting(
        "https://jobs.lever.co/hevodata/abc123",
        html,
        company="Manual Co",
        title="Manual Title",
        location="Remote India",
    )

    assert posting.company_name == "Manual Co"
    assert posting.title == "Manual Title"
    assert posting.location_raw == "Remote India"
