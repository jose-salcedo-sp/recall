#!/usr/bin/env python3
"""Seed Recall `chunks` with a demo personal-memory corpus.

Requires:
    pip install 'psycopg[binary]' httpx

Env:
    DATABASE_URL   default postgresql://recall:recall@localhost:5432/recall
    EMBEDDER_URL   default http://localhost:8081

Embeds `statement + " " + text` via POST {EMBEDDER_URL}/v1/embeddings
(OpenAI-compatible). Upserts into brain_id 00000000-0000-0000-0000-000000000001.
"""

from __future__ import annotations

import os
import sys
import uuid

BRAIN_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
ZERO_VEC_SQL = "array_fill(0::real, ARRAY[768])::vector"
EXPECTED_DIM = 768
BATCH = 32

# grantor brains (stable, for granted rows)
MAYA = uuid.UUID("00000000-0000-0000-0000-000000000002")
ANDRE = uuid.UUID("00000000-0000-0000-0000-000000000003")
PRIYA = uuid.UUID("00000000-0000-0000-0000-000000000004")
PATEL = uuid.UUID("00000000-0000-0000-0000-000000000005")
HR = uuid.UUID("00000000-0000-0000-0000-000000000006")
KEN = uuid.UUID("00000000-0000-0000-0000-000000000007")
ELENA = uuid.UUID("00000000-0000-0000-0000-000000000008")
LUIS = uuid.UUID("00000000-0000-0000-0000-000000000009")


def _id(n: int) -> uuid.UUID:
    return uuid.UUID(f"00000000-0000-4000-8000-{n:012d}")


def chunk(
    n: int,
    statement: str,
    text: str,
    *,
    origin: str = "personal",
    grantor_name: str | None = None,
    grantor_brain_id: uuid.UUID | None = None,
    sensitivity: str = "normal",
) -> dict:
    return {
        "id": _id(n),
        "statement": statement,
        "text": text,
        "origin": origin,
        "grantor_name": grantor_name,
        "grantor_brain_id": grantor_brain_id,
        "sensitivity": sensitivity,
    }


# Near-duplicate topic clusters are intentional: admission has to tell them apart.
CHUNKS: list[dict] = [
    chunk(1, "My sister Ana's birthday is March 14.",
          "Ana, my younger sister, was born on March 14. We usually call her in the morning."),
    chunk(2, "My brother Luis's birthday is July 2.",
          "Luis, my older brother, was born on July 2. He hates surprise parties."),
    chunk(3, "My sister Ana lives in Denver on York Street.",
          "Ana rents a walk-up on York Street in Denver, a few blocks from City Park."),
    chunk(4, "My mom's name is Carmen Salcedo.",
          "Carmen Salcedo is my mother. She still lives in the house I grew up in."),
    chunk(5, "Dad retired from the post office in 2019.",
          "My father retired from the USPS in 2019 after thirty-two years as a carrier."),
    chunk(6, "I'm allergic to penicillin.",
          "I have a documented penicillin allergy: hives and swelling. Do not take amoxicillin either."),
    chunk(7, "I'm allergic to peanuts and carry an EpiPen.",
          "Peanut allergy is anaphylactic. I keep an EpiPen in my bag and a spare in the kitchen."),
    chunk(8, "I get seasonal hay fever in April.",
          "Tree pollen wrecks me in April. Loratadine in the morning usually covers it."),
    chunk(9, "My blood type is O positive.",
          "Blood type O positive, confirmed at the last physical.",
          sensitivity="sensitive"),
    chunk(10, "I take 10mg of lisinopril each morning.",
          "Prescription: lisinopril 10mg once daily in the morning for blood pressure.",
          sensitivity="sensitive"),
    chunk(11, "The wifi password at the cabin is redmaple88.",
          "Cabin network is CabinNet. Password redmaple88, written on the router underside."),
    chunk(12, "Home wifi is Winterthorn-5G with password winterthorn.",
          "The apartment 5 GHz SSID is Winterthorn-5G. Password is winterthorn, lowercase."),
    chunk(13, "Office guest wifi password is VisitNexus2024.",
          "Guest SSID at the office is Nexus-Guest. Password VisitNexus2024."),
    chunk(14, "I signed the lease on the Oak Street apartment on 2024-06-01.",
          "Current home is the Oak Street apartment. Lease start date is 2024-06-01, 12 months."),
    chunk(15, "My previous place was 88 Pine Ave; the lease ended 2024-05-31.",
          "Before Oak Street I lived at 88 Pine Ave. That lease ended 2024-05-31."),
    chunk(16, "My dog Maple is a six-year-old border collie.",
          "Maple is a border collie, about six years old, high energy, knows wait and leave-it."),
    chunk(17, "Maple's vet is Dr. Singh at Riverside Animal.",
          "Maple sees Dr. Singh at Riverside Animal Clinic for annual vaccines and teeth."),
    chunk(18, "My cat Nimbus was adopted from the shelter on 2023-11-02.",
          "Nimbus, a gray domestic shorthair, came home from the county shelter on 2023-11-02."),
    chunk(19, "Nimbus only eats the chicken pâté, not the fish.",
          "Nimbus refuses fish wet food. Stick to the chicken pâté cans."),
    chunk(20, "Maple's microchip number is 981020000123456.",
          "Maple is chipped. HomeAgain number 981020000123456, registered to me."),
    chunk(21, "I work at Recall as a backend engineer.",
          "Job: backend engineer at Recall, mostly the retrieve-and-admit path."),
    chunk(22, "My manager is Elena Voss.",
          "Direct manager is Elena Voss. 1:1s are Tuesday mornings."),
    chunk(23, "My skip-level is Marcus Chen.",
          "Elena reports to Marcus Chen. Skip-levels are quarterly."),
    chunk(24, "I started this job on 2022-09-12.",
          "Recall start date is 2022-09-12. That is also my work anniversary."),
    chunk(25, "My work laptop hostname is recall-mbp-jose.",
          "Company MacBook hostname recall-mbp-jose. Asset tag on the underside."),
    chunk(26, "Checking account at First National ends in 4419.",
          "Primary checking is First National, account number last four 4419.",
          sensitivity="sensitive"),
    chunk(27, "The emergency fund target is $12,000.",
          "I am building a $12,000 emergency fund in the high-yield savings account."),
    chunk(28, "Rent is $2,150 due on the first.",
          "Oak Street rent is $2,150, auto-pay on the 1st to the landlord portal."),
    chunk(29, "My Visa credit card last four is 8821.",
          "Everyday Visa last four 8821. Used for travel and groceries.",
          sensitivity="sensitive"),
    chunk(30, "I max the 401k at 15% of paycheck.",
          "401k contribution is 15% of gross, which hits the annual limit most years."),
    chunk(31, "I drink coffee with oat milk, never dairy.",
          "Coffee order is always oat milk. Dairy upsets my stomach."),
    chunk(32, "I don't drink coffee after 2pm.",
          "No caffeine after 2pm or I will not sleep. Tea is fine in the evening."),
    chunk(33, "Favorite lunch is the banh mi from Lan's.",
          "Default weekday lunch: pork banh mi from Lan's on Colfax."),
    chunk(34, "I prefer window seats on flights.",
          "Always request a window seat. Aisle only if traveling with Maple in cabin."),
    chunk(35, "I sleep better when the room is at 67°F.",
          "Thermostat at night: 67°F. Warmer than that and I wake up."),
    chunk(36, "Passport expires 2028-04-19.",
          "US passport expires 2028-04-19. Renewal window opens a year before."),
    chunk(37, "Driver's license expires 2027-11-03.",
          "Colorado DL expires 2027-11-03. Real ID star is on it."),
    chunk(38, "TSA PreCheck number is 98123456.",
          "Known Traveler Number / TSA PreCheck is 98123456.",
          sensitivity="sensitive"),
    chunk(39, "GitHub username is jsalcedo.",
          "Personal GitHub is jsalcedo. Work repos are under the Recall org."),
    chunk(40, "The cabin lockbox code is 4218.",
          "Key lockbox on the cabin porch: code 4218. Relock when leaving."),
    chunk(41, "Last vacation was Lisbon in May 2025.",
          "May 2025 trip was a week in Lisbon with Maya. We walked everywhere."),
    chunk(42, "We stayed at Hotel do Chiado in Lisbon.",
          "Lisbon hotel was Hotel do Chiado, near the Baixa-Chiado metro."),
    chunk(43, "Next trip is a wedding in Austin on 2026-10-10.",
          "October 10, 2026: Lena's wedding in Austin. Flights not booked yet."),
    chunk(44, "I flew TAP flight TP88 to Lisbon.",
          "Outbound to Lisbon was TAP Air Portugal TP88 from DEN via LIS."),
    chunk(45, "I get motion sick on winding roads, not on planes.",
          "Carsick on switchbacks. Planes and trains are fine. Ginger chews in the glovebox."),
    chunk(46, "Partner Maya's coffee order is a cortado.",
          "Maya drinks a cortado, no extra foam. She will not finish a latte."),
    chunk(47, "Maya's birthday is December 5.",
          "Maya was born December 5. She likes quiet dinners, not big parties."),
    chunk(48, "Maya's work badge PIN is 3301.",
          "Maya asked me to remember her office badge PIN: 3301.",
          origin="granted", grantor_name="Maya", grantor_brain_id=MAYA,
          sensitivity="sensitive"),
    chunk(49, "Maya is allergic to shellfish.",
          "Maya has a shellfish allergy. No shrimp, crab, or oyster sauce.",
          origin="granted", grantor_name="Maya", grantor_brain_id=MAYA),
    chunk(50, "Maya's mom is named Rosa.",
          "Maya's mother is Rosa. She lives in Austin and hosts Thanksgiving."),
    chunk(51, "Building super is Andre, cell 555-0142.",
          "Landlord shared that the building super is Andre. Cell 555-0142 for leaks and heat.",
          origin="granted", grantor_name="Andre", grantor_brain_id=ANDRE),
    chunk(52, "Conference room B's HDMI adapter is in the left drawer.",
          "Elena noted the USB-C HDMI adapter for conference room B lives in the left drawer.",
          origin="granted", grantor_name="Elena Voss", grantor_brain_id=ELENA),
    chunk(53, "Last tetanus booster was 2021-08.",
          "Dr. Patel recorded a tetanus booster in August 2021. Next is due 2031.",
          origin="granted", grantor_name="Dr. Patel", grantor_brain_id=PATEL,
          sensitivity="sensitive"),
    chunk(54, "Priya's spare key is under the blue planter.",
          "Priya granted that her spare house key is under the blue planter by the steps.",
          origin="granted", grantor_name="Priya", grantor_brain_id=PRIYA),
    chunk(55, "Maya's sister Lena lives in Austin.",
          "Maya's sister Lena lives in Austin. That is whose wedding we are flying to.",
          origin="granted", grantor_name="Maya", grantor_brain_id=MAYA),
    chunk(56, "Employee ID is R-10428.",
          "HR listed my employee ID as R-10428. Use it on the benefits portal.",
          origin="granted", grantor_name="HR", grantor_brain_id=HR),
    chunk(57, "Next dental cleaning is booked for 2026-10-02.",
          "Dr. Okonkwo's office booked a cleaning on 2026-10-02 at 9:30am.",
          origin="granted", grantor_name="Dr. Okonkwo", grantor_brain_id=PATEL),
    chunk(58, "Trash pickup is Tuesday nights.",
          "Neighbor Ken said city trash and recycling go out Tuesday nights.",
          origin="granted", grantor_name="Ken", grantor_brain_id=KEN),
    chunk(59, "Luis's wedding anniversary is June 8.",
          "Luis asked me to remember his anniversary with Sofia: June 8.",
          origin="granted", grantor_name="Luis", grantor_brain_id=LUIS),
    chunk(60, "My childhood dog was named Puck.",
          "Before Maple we had a mutt named Puck. He lived to fourteen."),
    chunk(61, "I wear size 10.5 US shoes.",
          "Shoe size is 10.5 US / 44 EU. Wide in running shoes."),
    chunk(62, "I learned Spanish at home and English at school.",
          "First language at home was Spanish. English started in kindergarten."),
    chunk(63, "The car is a 2019 Subaru Outback, plate KRN-441.",
          "Daily driver: 2019 Subaru Outback, Colorado plate KRN-441, dark green."),
    chunk(64, "Auto insurance is State Farm policy 448-221.",
          "Car insurance is State Farm, policy 448-221, on the Outback.",
          sensitivity="sensitive"),
    chunk(65, "I keep the spare house key in the kitchen junk drawer.",
          "Spare Oak Street key is in the kitchen junk drawer, under the takeout menus."),
    chunk(66, "Gym is Midtown Iron, membership number 90021.",
          "I lift at Midtown Iron. Membership number 90021, barcode on the app."),
    chunk(67, "I quit smoking on 2018-01-01.",
          "Last cigarette was New Year's Day 2018. Do not offer me one."),
    chunk(68, "Favorite book is The Left Hand of Darkness.",
          "I reread Le Guin, The Left Hand of Darkness, every few years."),
    chunk(69, "My dentist is Dr. Okonkwo on Colfax.",
          "Dental home is Dr. Okonkwo on Colfax, two blocks west of Lan's."),
    chunk(70, "I keep a photocopy of my passport in the fire safe.",
          "Passport photocopy and the original live in the small fire safe in the closet."),
]


def _sql_str(value: str | None) -> str:
    if value is None:
        return "NULL"
    return "'" + value.replace("'", "''") + "'"


def _sql_uuid(value: uuid.UUID | None) -> str:
    if value is None:
        return "NULL"
    return f"'{value}'"


def emit_sql() -> str:
    lines = [
        "-- Demo corpus for Recall. Embeddings are placeholder zero-vectors;",
        "-- they must be backfilled by scripts/seed.py against a running embedder.",
        "",
        "INSERT INTO chunks (",
        "    id, brain_id, text, statement, embedding,",
        "    origin, grantor_brain_id, grantor_name, sensitivity",
        ") VALUES",
    ]
    values = []
    for c in CHUNKS:
        values.append(
            "    ("
            f"{_sql_uuid(c['id'])}, {_sql_uuid(BRAIN_ID)}, {_sql_str(c['text'])}, "
            f"{_sql_str(c['statement'])}, {ZERO_VEC_SQL}, {_sql_str(c['origin'])}, "
            f"{_sql_uuid(c['grantor_brain_id'])}, {_sql_str(c['grantor_name'])}, "
            f"{_sql_str(c['sensitivity'])})"
        )
    lines.append(",\n".join(values) + "")
    lines.append("ON CONFLICT (id) DO UPDATE SET")
    lines.append("    brain_id = EXCLUDED.brain_id,")
    lines.append("    text = EXCLUDED.text,")
    lines.append("    statement = EXCLUDED.statement,")
    lines.append("    origin = EXCLUDED.origin,")
    lines.append("    grantor_brain_id = EXCLUDED.grantor_brain_id,")
    lines.append("    grantor_name = EXCLUDED.grantor_name,")
    lines.append("    sensitivity = EXCLUDED.sensitivity,")
    lines.append("    updated_at = now();")
    lines.append("")
    return "\n".join(lines) + "\n"


def embed_batch(client, url: str, texts: list[str]) -> list[list[float]]:
    r = client.post(
        f"{url.rstrip('/')}/v1/embeddings",
        json={"input": texts, "model": "recall-embed"},
        timeout=60.0,
    )
    r.raise_for_status()
    payload = r.json()
    rows = sorted(payload["data"], key=lambda d: d["index"])
    if len(rows) != len(texts):
        print(f"embedder returned {len(rows)} vectors for {len(texts)} inputs", file=sys.stderr)
        sys.exit(1)
    out = []
    for i, row in enumerate(rows):
        vec = row["embedding"]
        if len(vec) != EXPECTED_DIM:
            print(
                f"embedder returned dim {len(vec)}, expected {EXPECTED_DIM} (input index {i})",
                file=sys.stderr,
            )
            sys.exit(1)
        out.append(vec)
    return out


def main() -> None:
    import httpx
    import psycopg

    db = os.environ.get("DATABASE_URL", "postgresql://recall:recall@localhost:5432/recall")
    embedder = os.environ.get("EMBEDDER_URL", "http://localhost:8081")

    texts = [f"{c['statement']} {c['text']}" for c in CHUNKS]
    vectors: list[list[float]] = []
    with httpx.Client() as client:
        for i in range(0, len(texts), BATCH):
            vectors.extend(embed_batch(client, embedder, texts[i : i + BATCH]))

    sql = """
        INSERT INTO chunks (
            id, brain_id, text, statement, embedding,
            origin, grantor_brain_id, grantor_name, sensitivity
        ) VALUES (
            %s, %s, %s, %s, %s::vector,
            %s, %s, %s, %s
        )
        ON CONFLICT (id) DO UPDATE SET
            brain_id = EXCLUDED.brain_id,
            text = EXCLUDED.text,
            statement = EXCLUDED.statement,
            embedding = EXCLUDED.embedding,
            origin = EXCLUDED.origin,
            grantor_brain_id = EXCLUDED.grantor_brain_id,
            grantor_name = EXCLUDED.grantor_name,
            sensitivity = EXCLUDED.sensitivity,
            updated_at = now()
    """
    with psycopg.connect(db) as conn:
        with conn.cursor() as cur:
            for c, vec in zip(CHUNKS, vectors, strict=True):
                cur.execute(
                    sql,
                    (
                        c["id"],
                        BRAIN_ID,
                        c["text"],
                        c["statement"],
                        "[" + ",".join(str(x) for x in vec) + "]",
                        c["origin"],
                        c["grantor_brain_id"],
                        c["grantor_name"],
                        c["sensitivity"],
                    ),
                )
        conn.commit()

    granted = sum(1 for c in CHUNKS if c["origin"] == "granted")
    print(f"upserted {len(CHUNKS)} chunks (brain_id={BRAIN_ID}, granted={granted})")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--emit-sql":
        sys.stdout.write(emit_sql())
    else:
        main()
