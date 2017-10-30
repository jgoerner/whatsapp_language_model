# Whatsapp Language Model
## What is it about
This repo contains a minor code snippet to generate (very, very) simple language models based on whatsapp 
conversation logs.
## How to use it
0. (optional) create a virtualenv to not mess up with your installed packages
1. [export the whatsapp conversation](https://faq.whatsapp.com/en/android/23756533) (**without**) media from your phone
2. install necessary dependencies `pip install -r requirements.txt`
3. train the language model
``` python
from wa_language_model import WhatsAppLanguageModel

WALM = WhatsAppLanguageModel("~/chat.txt")
```
4. generate a sentence
```python
WALM.generate_sentence(author="john doe")
```

Other neat attributes are `WALM.log` - which will show you the chat as a nicely formatted pandas Dataframe
and `WALM.authors` - which will give you the list of authors in that conversation.
## Further Improvements
- [ ] implement proper probabilites to introduce non-determinism in sequence generation
- [ ] implement <START> at the beginning of each sentence to get proper sentence beginnings
- [ ] implement flexible specification of n-gram range to generate sentences
- [ ] support other lang (beside German)
