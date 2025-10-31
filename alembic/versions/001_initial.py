"""Initial migration - create tables

Revision ID: 001_initial
Revises: 
Create Date: 2025-10-31 13:54:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create cameras table
    op.create_table('cameras',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('name', sa.String(length=128), nullable=False),
    sa.Column('url', sa.Text(), nullable=False),
    sa.Column('location', sa.String(length=256), nullable=True),
    sa.Column('active', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name')
    )
    
    # Create criminals table
    op.create_table('criminals',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('external_id', sa.String(length=64), nullable=True),
    sa.Column('name', sa.String(length=128), nullable=False),
    sa.Column('age', sa.Integer(), nullable=True),
    sa.Column('gender', sa.String(length=16), nullable=True),
    sa.Column('crime_type', sa.String(length=128), nullable=True),
    sa.Column('threat_level', sa.String(length=32), nullable=True),
    sa.Column('notes', sa.Text(), nullable=True),
    sa.Column('photo_path', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_criminals_external_id'), 'criminals', ['external_id'], unique=False)
    op.create_index(op.f('ix_criminals_name'), 'criminals', ['name'], unique=False)
    op.create_index(op.f('ix_criminals_crime_type'), 'criminals', ['crime_type'], unique=False)
    op.create_index(op.f('ix_criminals_threat_level'), 'criminals', ['threat_level'], unique=False)
    
    # Create embeddings table
    op.create_table('embeddings',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('criminal_id', sa.Integer(), nullable=False),
    sa.Column('vector', sa.Text(), nullable=False),
    sa.Column('dim', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['criminal_id'], ['criminals.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_embeddings_criminal_id'), 'embeddings', ['criminal_id'], unique=False)
    
    # Create incidents table
    op.create_table('incidents',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('camera_id', sa.Integer(), nullable=False),
    sa.Column('timestamp', sa.DateTime(), nullable=False),
    sa.Column('image_path', sa.Text(), nullable=True),
    sa.Column('face_count', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['camera_id'], ['cameras.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_incidents_camera_id'), 'incidents', ['camera_id'], unique=False)
    op.create_index(op.f('ix_incidents_timestamp'), 'incidents', ['timestamp'], unique=False)
    
    # Create incident_matches table
    op.create_table('incident_matches',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('incident_id', sa.Integer(), nullable=False),
    sa.Column('criminal_id', sa.Integer(), nullable=False),
    sa.Column('similarity', sa.Float(), nullable=False),
    sa.Column('bbox_x1', sa.Integer(), nullable=False),
    sa.Column('bbox_y1', sa.Integer(), nullable=False),
    sa.Column('bbox_x2', sa.Integer(), nullable=False),
    sa.Column('bbox_y2', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['criminal_id'], ['criminals.id'], ),
    sa.ForeignKeyConstraint(['incident_id'], ['incidents.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_incident_matches_criminal_id'), 'incident_matches', ['criminal_id'], unique=False)
    op.create_index(op.f('ix_incident_matches_incident_id'), 'incident_matches', ['incident_id'], unique=False)
    op.create_index(op.f('ix_incident_matches_similarity'), 'incident_matches', ['similarity'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_incident_matches_similarity'), table_name='incident_matches')
    op.drop_index(op.f('ix_incident_matches_incident_id'), table_name='incident_matches')
    op.drop_index(op.f('ix_incident_matches_criminal_id'), table_name='incident_matches')
    op.drop_table('incident_matches')
    
    op.drop_index(op.f('ix_incidents_timestamp'), table_name='incidents')
    op.drop_index(op.f('ix_incidents_camera_id'), table_name='incidents')
    op.drop_table('incidents')
    
    op.drop_index(op.f('ix_embeddings_criminal_id'), table_name='embeddings')
    op.drop_table('embeddings')
    
    op.drop_index(op.f('ix_criminals_threat_level'), table_name='criminals')
    op.drop_index(op.f('ix_criminals_crime_type'), table_name='criminals')
    op.drop_index(op.f('ix_criminals_name'), table_name='criminals')
    op.drop_index(op.f('ix_criminals_external_id'), table_name='criminals')
    op.drop_table('criminals')
    
    op.drop_table('cameras')
