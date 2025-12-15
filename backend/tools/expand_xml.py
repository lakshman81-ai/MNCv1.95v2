import argparse
import music21
import copy

def expand_xml(input_xml, output_xml, repeats=10):
    print(f"Reading {input_xml}...")
    score = music21.converter.parse(input_xml)

    # We want to repeat the measures in the part.
    # Assuming single part for simplicity (P1)
    part = score.parts[0]
    original_measures = list(part.getElementsByClass(music21.stream.Measure))

    print(f"Found {len(original_measures)} measures.")

    # Clear part? No, we append.
    # But music21 measures are tricky.
    # Better to create a new part and add measures.

    new_part = music21.stream.Part()
    new_part.id = part.id
    new_part.partName = part.partName

    # Copy initial attributes from first measure if needed (key, time, clef)
    # usually they are in the first measure.

    measure_count = 1

    for r in range(repeats):
        for m in original_measures:
            # Deep copy measure
            m_new = copy.deepcopy(m)
            m_new.number = measure_count

            # Remove attributes (clef, key, time) from subsequent repeats to avoid resetting?
            # Or keep them? Usually fine to keep them.
            # But we might want to ensure they are consistent.

            new_part.append(m_new)
            measure_count += 1

    new_score = music21.stream.Score()
    new_score.insert(0, new_part)

    print(f"Writing {output_xml} with {measure_count-1} measures...")
    new_score.write('musicxml', fp=output_xml)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input XML")
    parser.add_argument("output", help="Output XML")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    expand_xml(args.input, args.output, args.repeats)
