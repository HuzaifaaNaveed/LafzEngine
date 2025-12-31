# LafzEngine
A hybrid Urdu-to-Roman transliteration system that combines NER-aware processing, a gazette-based lookup, and a fine-tuned neural model to generate accurate and consistent Romanized text from Urdu.

### How to Run
1. Activate the environment
2. Download the model folder from the google drive and replace with current model folder.
3. Place your inputs in the inputs.txt folder
4. Go into the codes directory using "cd codes"
5. In the cmd, run python pipeline.py using "python pipeline.py" to run the entire pipeline
6. Run python tr_only.py using "python tr_only.py" to run the transliteration only
7. Running this will give you the results in the outputs.txt file
8. Then run check.py using "python check.py"
9. This will check the transliterations, and log everything to the logs.json file
