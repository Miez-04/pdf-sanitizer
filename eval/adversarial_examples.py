"""
eval/adversarial_examples.py

Every sentence here was written by hand, specifically to NOT resemble
anything data/build_corpus.py could have generated:
- None use the 3 fixed connector-clause templates per entity type
  (build_corpus.py's actual limitation, documented earlier).
- Entities appear mid-sentence, not just as the object of "at"/
  "contact"/"located at" — the exact positions the templates always
  use.
- Multiple registers: formal memo, WhatsApp-casual Manglish, invoice,
  résumé bullet, government form (with DIFFERENT field labels than
  training data — "Waris:", "Saksi:", "Pemilik:", not "Nama penuh:"),
  news attribution, fully negative (no PII at all).
- Deliberately malformed/edge-case entity formats: NRIC without
  dashes, NRIC with spaced dashes, phone with dot separators, phone
  with an extension — some of these are EXPECTED to fail (regex was
  never built to catch dot-separated phones), and that's the point:
  an adversarial set should show real failure modes, not just wins.
- Decoy numbers (invoice refs, prices, dates) that are NOT PII, to
  check false-positive rate, not just recall.
- Multiple entities of the same type in one sentence (two witnesses).

Entity spans are marked inline as {TYPE:text}, parsed by
build_adversarial_set.py — this keeps the authored sentences readable
while still producing exact, error-free IOB2 tags.
"""

ADVERSARIAL_SENTENCES = [
    # --- Formal memo, entity mid-sentence, not sentence-initial ---
    "Following the internal audit, the department head {PERSON:Rosnah binti Kamal} was asked to submit a written explanation by Friday.",
    "The complaint filed against the vendor was escalated to {PERSON:Mr. Devendran Krishnasamy}, who oversees the procurement division.",

    # --- WhatsApp-style Manglish, casual ---
    "eh can whatsapp {PERSON:Ah Chong} lah his number {PHONE:016-2345678} he sure pick up one",
    "confirm tomorrow meet up at {ADDRESS:Lot 15, Jalan SS2/24, 47300 Petaling Jaya} ya dont late",

    # --- Invoice / receipt style, multiple entity types packed together ---
    "Invoice billed to {PERSON:Encik Faizal bin Rahim}, delivery address {ADDRESS:No. 8, Jalan Meranti 3, Taman Perindustrian, 81100 Johor Bahru, Johor}, contact {PHONE:07-3456789}.",

    # --- Resume / CV bullet ---
    "References available upon request; primary referee is {PERSON:Dr. Chandra Mohan a/l Sivalingam}, reachable at {PHONE:+60123456789}.",

    # --- Government form, DIFFERENT field labels than training data ---
    "Waris : {PERSON:Halimah binti Yusof} , No. K/P : {NRIC:850211-10-5678}",
    "Saksi pertama : {PERSON:Tan Ah Kow} , Saksi kedua : {PERSON:Muthu a/l Ramasamy}",
    "Pemilik berdaftar bagi hartanah ini ialah {PERSON:Lim Poh Choo}.",

    # --- News-style quote attribution mid-sentence ---
    "\"We will investigate the matter thoroughly,\" said {PERSON:Assistant Commissioner Zulhelmi Abdullah} during the press conference yesterday.",

    # --- Malformed / edge-case entity formats (some EXPECTED to fail) ---
    "IC: {NRIC:890523125566}",  # no dashes at all
    "NRIC No : {NRIC:79 02 14 - 08 - 3321}",  # spaced-out dashes
    "You can dial {PHONE:012.345.6789} anytime after 6pm.",  # dot separators, not dash/space
    "Office line {PHONE:03-77281234} ext. 205 for the finance team.",  # extension suffix

    # --- Address styles the templates never covered ---
    "Berhampiran dengan stesen LRT, unit tersebut terletak di {ADDRESS:Tingkat 9, Blok C, Menara Uncang Emas, Jalan Loke Yew, 55200 Kuala Lumpur}.",
    "Kedai runcit itu beroperasi dari {ADDRESS:PT 2291, Kampung Baru Subang, 40150 Shah Alam, Selangor} sejak tahun 1998.",

    # --- Decoy numbers: NOT PII, testing false-positive rate ---
    "The shipment reference number is INV-2024-00981, unrelated to any personal data.",
    "Total amount payable is RM 3,450.00 before the 15th of next month.",
    "The building was constructed in 1998 and renovated again in 2015.",

    # --- Fully negative: no PII at all ---
    "The weather forecast predicts scattered thunderstorms across the Klang Valley this afternoon.",
    "Kerajaan negeri mengumumkan bantuan banjir tambahan bagi kawasan terjejas.",
    "Traffic congestion was reported along the federal highway due to ongoing roadworks.",

    # --- Multiple same-type entities in one sentence ---
    "Both {PERSON:Farah Adila binti Zainal} and {PERSON:Kavitha a/p Ganesan} signed the agreement as joint tenants.",

    # --- Decoy capitalized non-person proper nouns near a real PERSON ---
    "According to Bernama, {PERSON:Puan Sri Zaleha Osman} chaired the Petronas charity gala held last week.",

    # --- Entity buried deep in a long, run-on sentence ---
    "After several rounds of negotiation that lasted nearly three hours, the two parties finally agreed that the property, previously registered under {PERSON:Ahmad Zaki bin Ibrahim}, would be transferred with immediate effect once the outstanding balance is cleared.",

    # --- Code-mixed sentence with entity inline ---
    "So the plan is, {PERSON:Jessica Wong} akan hantar dokumen tu ke {ADDRESS:Suite 12-3, Menara Hap Seng, Jalan P Ramlee, 50250 Kuala Lumpur} by Thursday.",
]
