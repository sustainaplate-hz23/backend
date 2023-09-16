import os
from difflib import SequenceMatcher
import re
import json
from modules.llm import BamLLM
from llama_index import ServiceContext
from llama_index.storage.storage_context import StorageContext, SimpleDocumentStore, SimpleIndexStore, SimpleVectorStore
from llama_index import load_index_from_storage
from llama_index.retrievers import VectorIndexRetriever
import pandas as pd


class MigrosRetriever:
    def __init__(self):
        llm = BamLLM()
        self.sc = ServiceContext.from_defaults(
            llm=llm,
            context_window=int(os.getenv('CONTEXT_WINDOW')),
            num_output=int(os.getenv('MAX_OUTPUT_TOKENS'))
        )
        self.query_engine = None
        self.index = None

    def load_index(self):
        storage_context = StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(persist_dir='modules/migros_data_v3'),
            vector_store=SimpleVectorStore.from_persist_dir(persist_dir='modules/migros_data_v3'),
            index_store=SimpleIndexStore.from_persist_dir(persist_dir='modules/migros_data_v3'),
        )
        index = load_index_from_storage(
            storage_context=storage_context,
            service_context=self.sc
        )
        self.index = index

    @staticmethod
    def word_percentage_in_string(recipe_ing, user_ing):
        words_recipe_ing = recipe_ing.split()
        words_user_ing = user_ing.split()

        matching_words_count = 0

        for word_recipe_ing in words_recipe_ing:
            for word_user_ing in words_user_ing:
                if SequenceMatcher(None, word_recipe_ing, word_user_ing).ratio() >= 0.9:
                    matching_words_count += 1
                    break

        percentage = (matching_words_count / max(len(words_recipe_ing), len(words_user_ing)))
        if percentage >= 2/3:
            return True
        return False

    def post_process_results(self, docs, user_ingredients):
        final_recipes_df = pd.read_csv('modules/data/final_recipes.csv')
        with open('modules/data/found_ingredients_dict.json', 'r') as fp:
            found_ingredients_dict = json.load(fp)

        matched_recipe_ids = []
        for doc in docs:
            # Define the regex pattern to match the Recipe ID
            pattern = r"Recipe ID:\s+(\d+)"

            # Use re.search to find the first match of the pattern in the text
            match = re.search(pattern, doc.text)

            # Extract the Recipe ID if a match is found
            if match:
                recipe_id = match.group(1)
                matched_recipe_ids.append(int(recipe_id))
                print("Recipe ID:", recipe_id)
            else:
                print("Recipe ID not found in the text.")

        filtered_recipes_df = final_recipes_df.loc[final_recipes_df['id'].isin(matched_recipe_ids)]
        chosen_recipes = []

        for recipe in filtered_recipes_df.itertuples():
            recipe_ingredients = eval(recipe.ingredients)
            remove_list = []
            for recipe_ing in recipe_ingredients:
                for user_ing in user_ingredients:
                    if self.word_percentage_in_string(recipe_ing, user_ing):
                        remove_list.append(recipe_ing)
            # remove_list += ['salt', 'pepper']
            new_ings = [elem for elem in eval(recipe.ingredients) if elem not in remove_list]
            sustainability = sum([found_ingredients_dict[key]["sustainability_rating"]
                                  if key in found_ingredients_dict else 6 for key in new_ings ])

            if len(recipe_ingredients) != len(new_ings):
                chosen_recipes.append((recipe, new_ings, sustainability))

        chosen_recipes = sorted(chosen_recipes, key=lambda x: x[-1])

        return chosen_recipes

    @staticmethod
    def recipe_to_str(recipe_doc):
        ingredients_to_buy = recipe_doc[1]
        recipe = {
            "name": recipe_doc[0].name,
            "description": str(recipe_doc[0].description),
            "ingredients": eval(recipe_doc[0].ingredients),
            "minutes_to_cook": recipe_doc[0].minutes,
            "steps": eval(recipe_doc[0].steps),
            "tags": eval(recipe_doc[0].tags)
        }
        sustainability_score = recipe_doc[2]
        return {
            "ingredients_to_buy": ingredients_to_buy,
            "recipe_info": recipe,
            "sustainability_score": sustainability_score
        }

    def query(self, ingredients):
        prompt = f"""
        Ingredients: {", ".join(ingredients)}
        """

        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=100,
        )

        docs = retriever.retrieve(prompt)
        docs = self.post_process_results(docs, ingredients)

        final_results = []
        for doc in docs:
            o = self.recipe_to_str(doc)
            final_results.append(o)

        return final_results
