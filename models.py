from sqlalchemy import Column, Integer, String, MetaData, Table

metadata = MetaData()

def create_table(table_name):
    return Table(
        table_name, metadata,
        Column('id', Integer, primary_key=True),
        Column('filename', String(255), nullable=False),
        extend_existing=True
    )

images = create_table('images')
animal = create_table('animal')
etc = create_table('etc')
food = create_table('food')
human = create_table('human')
nature = create_table('nature')
place = create_table('place')