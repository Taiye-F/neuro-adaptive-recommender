"""Initial baseline with users, child_profiles, assessments, recommendations

Revision ID: 5527db5388e3
Revises: 
Create Date: 2026-09-09 19:23:22.237393

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '5527db5388e3'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('username', sa.String(), nullable=False),
        sa.Column('email', sa.String(), nullable=False),
        sa.Column('hashed_password', sa.String(), nullable=False),
        sa.Column('role', sa.String(), nullable=False, server_default='parent'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('subscription_tier', sa.String(), nullable=False, server_default='free'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('refresh_token', sa.String(), nullable=True),
    )
    op.create_index('ix_users_id', 'users', ['id'])
    op.create_index('ix_users_username', 'users', ['username'], unique=True)
    op.create_index('ix_users_email', 'users', ['email'], unique=True)

    # 2. child_profiles table
    op.create_table(
        'child_profiles',
        sa.Column('id', sa.Uuid(), nullable=False, primary_key=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('first_name', sa.String(length=50), nullable=False),
        sa.Column('date_of_birth', sa.Date(), nullable=False),
        sa.Column('biological_sex', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_child_profiles_id', 'child_profiles', ['id'])
    op.create_index('ix_child_profiles_user_id', 'child_profiles', ['user_id'])

    # 3. screening_assessments table
    op.create_table(
        'screening_assessments',
        sa.Column('id', sa.Uuid(), nullable=False, primary_key=True),
        sa.Column('child_id', sa.Uuid(), sa.ForeignKey('child_profiles.id', ondelete='CASCADE'), nullable=False),
        sa.Column('age_months', sa.Integer(), nullable=False),
        sa.Column('a1_score', sa.Integer(), nullable=False),
        sa.Column('a2_score', sa.Integer(), nullable=False),
        sa.Column('a3_score', sa.Integer(), nullable=False),
        sa.Column('a4_score', sa.Integer(), nullable=False),
        sa.Column('a5_score', sa.Integer(), nullable=False),
        sa.Column('a6_score', sa.Integer(), nullable=False),
        sa.Column('a7_score', sa.Integer(), nullable=False),
        sa.Column('a8_score', sa.Integer(), nullable=False),
        sa.Column('a9_score', sa.Integer(), nullable=False),
        sa.Column('a10_score', sa.Integer(), nullable=False),
        sa.Column('risk_probability', sa.Float(), nullable=False),
        sa.Column('is_high_risk', sa.Boolean(), nullable=False),
        sa.Column('total_flags', sa.Integer(), nullable=False),
        sa.Column('profile_text', sa.Text(), nullable=False),
        sa.Column('profile_explained', sa.Text(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_screening_assessments_id', 'screening_assessments', ['id'])
    op.create_index('ix_screening_assessments_child_id', 'screening_assessments', ['child_id'])
    op.create_index('ix_screening_assessments_completed_at', 'screening_assessments', ['completed_at'])

    # 4. assessment_recommendations table
    op.create_table(
        'assessment_recommendations',
        sa.Column('id', sa.Uuid(), nullable=False, primary_key=True),
        sa.Column('assessment_id', sa.Uuid(), sa.ForeignKey('screening_assessments.id', ondelete='CASCADE'), nullable=False),
        sa.Column('resource_type', sa.String(length=20), nullable=False),
        sa.Column('item_name', sa.String(length=255), nullable=False),
        sa.Column('category', sa.String(length=100), nullable=True),
        sa.Column('match_score', sa.Float(), nullable=False),
        sa.Column('rank', sa.Integer(), nullable=False),
    )
    op.create_index('ix_assessment_recommendations_id', 'assessment_recommendations', ['id'])
    op.create_index('ix_assessment_recommendations_assessment_id', 'assessment_recommendations', ['assessment_id'])


def downgrade() -> None:
    op.drop_table('assessment_recommendations')
    op.drop_table('screening_assessments')
    op.drop_table('child_profiles')
    op.drop_table('users')
