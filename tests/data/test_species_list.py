from pathlib import Path

from tiberius_orf.data.species_list import (
    Species,
    parse_species_table,
    write_csvs,
)


FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def test_parse_minimal_fixture():
    species = parse_species_table(FIXTURES / "insecta_data_info_mini.tex")

    by_split = {s.split: [] for s in species}
    for s in species:
        by_split[s.split].append(s)

    assert len(by_split["training"]) == 3
    assert len(by_split["test"]) == 1
    # Drosophila melanogaster is in both test and val in the fixture;
    # parser must drop it from val, leaving only Tenebrio molitor.
    assert len(by_split["val"]) == 1
    assert by_split["val"][0].name == "Tenebrio molitor"

    aedes = next(s for s in species if s.name == "Aedes aegypti")
    assert aedes == Species(
        name="Aedes aegypti",
        accession="GCF_002204515.2",
        annotation="RefSeq",
        split="training",
    )
    assert aedes.underscored == "Aedes_aegypti"

    dmel = next(s for s in species if s.name == "Drosophila melanogaster")
    assert dmel.split == "test"
    assert dmel.accession == "GCF_000001215.4"


def test_write_csvs_roundtrip(tmp_path):
    species = parse_species_table(FIXTURES / "insecta_data_info_mini.tex")
    paths = write_csvs(species, tmp_path)

    assert set(paths) == {"training", "val", "test"}

    training_csv = paths["training"].read_text().strip().splitlines()
    assert training_csv[0] == "species,accession,annotation"
    assert len(training_csv) == 1 + 3  # header + 3 training species
    assert any("Aedes aegypti,GCF_002204515.2,RefSeq" in line for line in training_csv)
