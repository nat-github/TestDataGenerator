import pandas as pd
from typing import Dict, List, Set, Any
from models.config_models import RelationshipConfig


class DataValidator:
    def __init__(self):
        self.validation_report = {}

    def validate_relationships(self, data: Dict[str, pd.DataFrame],
                               relationships: List[RelationshipConfig]) -> bool:
        """Validate that all foreign key relationships are maintained"""
        self.validation_report = {
            'relationships': [],
            'valid_count': 0,
            'invalid_count': 0,
            'details': []
        }

        all_valid = True

        for relationship in relationships:
            source_table = relationship.source_table
            target_table = relationship.target_table

            if source_table in data and target_table in data:
                source_data = data[source_table]
                target_data = data[target_table]

                if (relationship.source_column in source_data.columns and
                        relationship.target_column in target_data.columns):

                    is_valid = self._validate_single_relationship(
                        source_data, target_data, relationship
                    )

                    if is_valid:
                        self.validation_report['valid_count'] += 1
                    else:
                        self.validation_report['invalid_count'] += 1
                        all_valid = False

                else:
                    self.validation_report['details'].append({
                        'relationship': f"{source_table}.{relationship.source_column} → {target_table}.{relationship.target_column}",
                        'status': 'ERROR',
                        'message': 'Required columns not found'
                    })
                    all_valid = False

        return all_valid

    def _validate_single_relationship(self, source_data: pd.DataFrame,
                                      target_data: pd.DataFrame,
                                      relationship: RelationshipConfig) -> bool:
        """Validate a single foreign key relationship"""
        source_values = set(source_data[relationship.source_column].dropna().unique())
        target_values = set(target_data[relationship.target_column].dropna().unique())

        invalid_values = source_values - target_values
        is_valid = len(invalid_values) == 0

        relationship_info = {
            'relationship': f"{relationship.source_table}.{relationship.source_column} → {relationship.target_table}.{relationship.target_column}",
            'status': 'VALID' if is_valid else 'INVALID',
            'source_values_count': len(source_values),
            'target_values_count': len(target_values),
            'invalid_values_count': len(invalid_values),
            'invalid_values_sample': list(invalid_values)[:5] if invalid_values else []
        }

        self.validation_report['relationships'].append(relationship_info)
        self.validation_report['details'].append(relationship_info)

        return is_valid

    def get_validation_report(self) -> Dict[str, Any]:
        """Get detailed validation report"""
        return self.validation_report
