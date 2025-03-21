import asyncio
from ..database import db_manager


async def update_all_embeddings():
    """Update embeddings for all tables"""
    await db_manager.update_embeddings_for_table('clinical_trials_feed')
    await db_manager.update_embeddings_for_table('publications_feed')


if __name__ == "__main__":
    asyncio.run(update_all_embeddings())