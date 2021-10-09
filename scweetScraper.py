from Scweet.scweet import scrape



data = scrape(since="2021-10-05", until=None, from_account = 'elonmusk', interval=1, 
              headless=False, display_type="Latest", save_images=False, 
              resume=False, filter_replies=False, proximity=False)



