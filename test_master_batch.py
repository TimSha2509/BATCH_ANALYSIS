import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import master_batch as batch


def material_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Material_ID": [
                "26061XX",
                "DOE_Glycerol_60",
                "DOE_Glycerol_87",
                "DOE_Glycerol_98",
            ],
            "plasticizer": ["Glycerol", "Glycerol", "Glycerol", "Glycerol"],
            "concentration_plasticizer": [90.0, 60.0, 87.0, 98.5],
            "surface tension_mNm": [pd.NA, 58.0, 47.0, 44.0],
            "density_kg_l": [pd.NA, 1.15, 1.2, 1.25],
            "viscosity_ref_40_1s": [pd.NA, 10.0, 110.0, 1039.0],
        }
    )


def nozzle_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Nozzle_ID": ["2.5mm_A", "1mm_P", "G6_A"],
            "inner_diameter_mm": [1.8, pd.NA, 3.4],
        }
    )


def create_combo(
    root: Path,
    run_name: str,
    combo_name: str,
    metadata: dict[str, object],
) -> tuple[Path, Path]:
    run_folder = root / run_name
    combo_folder = run_folder / 'steps' / combo_name
    (combo_folder / 'images').mkdir(parents=True)
    with (combo_folder / 'step.json').open('w', encoding='utf-8') as file:
        json.dump(metadata, file)
    return run_folder, combo_folder


def combo_metadata(
    *,
    material_id: object = '26061XX',
    frequency: object = 100,
    nozzle_id: object = '2.5mm_A',
    mass_flow: object = 100,
    displacement: object = 10,
) -> dict[str, object]:
    return {
        'Material_ID': material_id,
        'set_frequency_hz': frequency,
        'nozzle_id': nozzle_id,
        'target_mass_flow_g_per_h': mass_flow,
        'target_displacement_um': displacement,
    }


class LevelIndexTests(unittest.TestCase):
    def test_displacement_labels_cover_one_through_five_levels(self) -> None:
        expected = {
            1: ['L'],
            2: ['L', 'H'],
            3: ['L', 'M', 'H'],
            4: ['L', 'M', 'H', 'H_2'],
            5: ['L', 'M', 'H', 'H_2', 'H_3'],
        }

        for level_count, labels in expected.items():
            with self.subTest(level_count=level_count):
                self.assertEqual(
                    [
                        batch.displacement_index_for_rank(rank, level_count)
                        for rank in range(level_count)
                    ],
                    labels,
                )

    def test_lookup_groups_normalized_metadata_and_skips_invalid_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            combo_folders: list[tuple[Path, Path]] = []

            def add(name: str, metadata: dict[str, object]) -> None:
                combo_folders.append(create_combo(root, 'run_a', name, metadata))

            for index, displacement in enumerate([10, 20, 30, 40, 50], start=1):
                add(f'combo_base_{index}', combo_metadata(displacement=displacement))
            add(
                'combo_duplicate',
                combo_metadata(material_id='Glycerol_90', displacement=10.0),
            )

            variants = {
                'combo_material': combo_metadata(
                    material_id='DOE_Glycerol_60',
                    displacement=999,
                ),
                'combo_frequency': combo_metadata(frequency=200, displacement=999),
                'combo_nozzle': combo_metadata(nozzle_id='G6_A', displacement=999),
                'combo_flow': combo_metadata(mass_flow=200, displacement=999),
            }
            for name, metadata in variants.items():
                add(name, metadata)

            invalid = {
                'combo_bad_material': combo_metadata(material_id='unknown'),
                'combo_bad_frequency': combo_metadata(frequency='invalid'),
                'combo_bad_nozzle': combo_metadata(nozzle_id=''),
                'combo_bad_flow': combo_metadata(mass_flow=None),
                'combo_bad_displacement': combo_metadata(displacement='invalid'),
            }
            for name, metadata in invalid.items():
                add(name, metadata)

            lookup = batch.build_displacement_index_lookup(
                combo_folders,
                materials=material_catalog(),
            )

        self.assertEqual(
            [
                lookup[('run_a', f'combo_base_{index}')]
                for index in range(1, 6)
            ],
            ['L', 'M', 'H', 'H_2', 'H_3'],
        )
        self.assertEqual(lookup[('run_a', 'combo_duplicate')], 'L')
        for name in variants:
            self.assertEqual(lookup[('run_a', name)], 'L')
        for name in invalid:
            self.assertNotIn(('run_a', name), lookup)

    def test_index_application_places_columns_and_is_idempotent(self) -> None:
        result = pd.DataFrame(
            {
                'run_folder': ['run_a'],
                'combo_folder': ['combo_1'],
                'target_displacement_um': [20],
                'set_frequency_hz': [100],
            }
        )
        indexed = batch.apply_frequency_index_lookup(
            result,
            {('run_a', 'combo_1'): 'M'},
        )
        indexed = batch.apply_displacement_index_lookup(
            indexed,
            {('run_a', 'combo_1'): 'H'},
        )

        self.assertEqual(indexed.loc[0, 'Frequency_index'], 'M')
        self.assertEqual(indexed.loc[0, 'Displacement_index'], 'H')
        self.assertEqual(
            indexed.columns.get_loc('Frequency_index'),
            indexed.columns.get_loc('set_frequency_hz') + 1,
        )
        self.assertEqual(
            indexed.columns.get_loc('Displacement_index'),
            indexed.columns.get_loc('target_displacement_um') + 1,
        )
        reapplied = batch.apply_displacement_index_lookup(
            indexed,
            {('run_a', 'combo_1'): 'H'},
        )
        self.assertTrue(reapplied.equals(indexed))
        self.assertEqual(
            batch.ALL_RUNS_COLUMNS.index('Displacement_index'),
            batch.ALL_RUNS_COLUMNS.index('target_displacement_um') + 1,
        )
        per_combo = batch.add_metadata_columns(
            pd.DataFrame({'particle_id': [1]}),
            {
                'target_displacement_um': 20,
                'Displacement_index': 'H',
            },
        )
        self.assertEqual(
            per_combo.columns.get_loc('Displacement_index'),
            per_combo.columns.get_loc('target_displacement_um') + 1,
        )

    def test_refresh_relabels_existing_rows_after_new_measurements(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_root = root / 'csv'
            csv_root.mkdir()
            combos = [
                create_combo(
                    root,
                    'run_a',
                    'combo_1',
                    combo_metadata(frequency=100, displacement=20),
                ),
                create_combo(
                    root,
                    'run_a',
                    'combo_2',
                    combo_metadata(frequency=100, displacement=40),
                ),
            ]

            def write_combo_csv(
                run_folder: Path,
                combo_folder: Path,
                metadata: dict[str, object],
            ) -> Path:
                csv_path = csv_root / f'{run_folder.name}__{combo_folder.name}.csv'
                pd.DataFrame(
                    [
                        {
                            'run_folder': run_folder.name,
                            'combo_folder': combo_folder.name,
                            'experiment_id': '',
                            **metadata,
                            'ECD_mm': 1.0,
                        }
                    ]
                ).to_csv(csv_path, index=False)
                return csv_path

            combo_1_csv = write_combo_csv(
                *combos[0],
                combo_metadata(frequency=100, displacement=20),
            )
            write_combo_csv(
                *combos[1],
                combo_metadata(frequency=100, displacement=40),
            )

            initial_frequency = batch.build_frequency_index_lookup(combos, csv_root)
            initial_displacement = batch.build_displacement_index_lookup(
                combos,
                csv_root,
                material_catalog(),
            )
            updated = batch.update_level_indices_in_existing_csvs(
                csv_root,
                initial_frequency,
                initial_displacement,
                material_catalog(),
                nozzle_catalog(),
            )
            self.assertEqual(updated, 2)
            self.assertEqual(
                batch.update_level_indices_in_existing_csvs(
                    csv_root,
                    initial_frequency,
                    initial_displacement,
                    material_catalog(),
                    nozzle_catalog(),
                ),
                0,
            )

            new_combos = [
                create_combo(
                    root,
                    'run_a',
                    'combo_3',
                    combo_metadata(frequency=100, displacement=10),
                ),
                create_combo(
                    root,
                    'run_a',
                    'combo_4',
                    combo_metadata(frequency=50, displacement=20),
                ),
            ]
            combos.extend(new_combos)
            write_combo_csv(
                *new_combos[0],
                combo_metadata(frequency=100, displacement=10),
            )
            write_combo_csv(
                *new_combos[1],
                combo_metadata(frequency=50, displacement=20),
            )

            refreshed_frequency = batch.build_frequency_index_lookup(combos, csv_root)
            refreshed_displacement = batch.build_displacement_index_lookup(
                combos,
                csv_root,
                material_catalog(),
            )
            batch.update_level_indices_in_existing_csvs(
                csv_root,
                refreshed_frequency,
                refreshed_displacement,
                material_catalog(),
                nozzle_catalog(),
            )
            refreshed_combo_1 = pd.read_csv(combo_1_csv)

        self.assertEqual(refreshed_combo_1.loc[0, 'Frequency_index'], 'M')
        self.assertEqual(refreshed_combo_1.loc[0, 'Displacement_index'], 'M')


class MaterialEnrichmentTests(unittest.TestCase):
    def test_canonicalizes_legacy_ids_and_refreshes_properties(self) -> None:
        result = pd.DataFrame(
            {
                "experiment_id": ["a", "b", "c", "d", "ipa"],
                "Material_ID": [
                    "Glycerol_60",
                    "glycerol_87",
                    "DOE_Glycerol_98",
                    "Glycerol_90",
                    "",
                ],
                "surface tension_mNm": [999.0] * 5,
                "density_kg_l": [999.0] * 5,
                "viscosity_ref_40_1s": [999.0] * 5,
            }
        )

        enriched = batch.enrich_material_properties(result, material_catalog())

        self.assertEqual(
            enriched["Material_ID"].tolist(),
            [
                "DOE_Glycerol_60",
                "DOE_Glycerol_87",
                "DOE_Glycerol_98",
                "26061XX",
                "",
            ],
        )
        self.assertEqual(
            enriched["surface tension_mNm"].iloc[:3].tolist(),
            [58.0, 47.0, 44.0],
        )
        self.assertTrue(enriched.loc[3:, "surface tension_mNm"].isna().all())
        self.assertEqual(
            enriched.columns.tolist()[1:5],
            ["Material_ID", *batch.MATERIAL_PROPERTY_COLUMNS],
        )
        self.assertTrue(
            batch.enrich_material_properties(enriched, material_catalog()).equals(enriched)
        )

    def test_unknown_non_empty_id_fails_but_blank_is_allowed(self) -> None:
        blank = batch.enrich_material_properties(
            pd.DataFrame({"Material_ID": [""]}),
            material_catalog(),
        )
        self.assertEqual(blank.loc[0, "Material_ID"], "")

        with self.assertRaisesRegex(ValueError, "Unknown"):
            batch.enrich_material_properties(
                pd.DataFrame({"Material_ID": ["Unknown"]}),
                material_catalog(),
            )

    def test_enrichment_is_idempotent_after_csv_round_trip(self) -> None:
        enriched = batch.enrich_material_properties(
            pd.DataFrame(
                {
                    "experiment_id": ["TEST_IPA", "Glycerol_90"],
                    "Material_ID": ["", "Glycerol_90"],
                }
            ),
            material_catalog(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "result.csv"
            enriched.to_csv(path, index=False)
            reloaded = pd.read_csv(path)

        refreshed = batch.enrich_material_properties(reloaded, material_catalog())
        self.assertTrue(batch.dataframes_equal_for_csv(refreshed, reloaded))

    def test_catalog_loader_rejects_missing_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "Materials.csv"
            pd.DataFrame({"Material_ID": ["A"]}).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "missing required columns"):
                batch.load_material_catalog(path)

    def test_catalog_loader_rejects_duplicate_ids_case_insensitively(self) -> None:
        catalog = material_catalog()
        catalog = pd.concat([catalog, catalog.iloc[[1]].assign(Material_ID="doe_glycerol_60")])

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "Materials.csv"
            catalog.to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "duplicate Material_ID"):
                batch.load_material_catalog(path)


class DimensionlessDiameterTests(unittest.TestCase):
    def test_calculates_ratio_and_leaves_unresolvable_rows_blank(self) -> None:
        result = pd.DataFrame(
            {
                "nozzle_id": ["2.5mm_A", "g6_a", "4mm_A", "", "1mm_P"],
                "ECD_mm": [3.6, 6.8, 4.0, 2.0, 1.0],
                "dd/di": [999.0] * 5,
            }
        )

        enriched = batch.enrich_dimensionless_diameter(result, nozzle_catalog())

        self.assertEqual(enriched["dd/di"].iloc[:2].tolist(), [2.0, 2.0])
        self.assertTrue(enriched["dd/di"].iloc[2:].isna().all())
        self.assertEqual(
            enriched.columns.get_loc("dd/di"),
            enriched.columns.get_loc("ECD_mm") + 1,
        )

    def test_ratio_is_idempotent_after_csv_round_trip(self) -> None:
        enriched = batch.enrich_dimensionless_diameter(
            pd.DataFrame(
                {
                    "nozzle_id": ["2.5mm_A", "4mm_A"],
                    "ECD_mm": [1.2345678901234567, 1.0],
                }
            ),
            nozzle_catalog(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "result.csv"
            enriched.to_csv(path, index=False)
            reloaded = pd.read_csv(path)

        refreshed = batch.enrich_dimensionless_diameter(reloaded, nozzle_catalog())
        self.assertTrue(batch.dataframes_equal_for_csv(refreshed, reloaded))

    def test_nozzle_catalog_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "Nozzles.csv"
            pd.DataFrame({"Nozzle_ID": ["A"]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                batch.load_nozzle_catalog(path)

            pd.DataFrame(
                {
                    "Nozzle_ID": ["A", "a"],
                    "inner_diameter_mm": [1.0, 1.0],
                }
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate Nozzle_ID"):
                batch.load_nozzle_catalog(path)

            pd.DataFrame(
                {"Nozzle_ID": ["A"], "inner_diameter_mm": [0.0]}
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "greater than zero"):
                batch.load_nozzle_catalog(path)


if __name__ == "__main__":
    unittest.main()
