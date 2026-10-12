"""Seeded synthetic corpus with topic-labelled queries.

Every document belongs to exactly one topic, and a query is relevant to every
document of its topic. Paragraphs open with a unique lead-in so two documents
never produce identical chunk text by accident; identical chunks would let
content deduplication flatter the incremental strategy.
"""

import random
from dataclasses import dataclass

from ..hashing import hash_bytes, hash_payload

TOPICS = {
    "cpu-scheduling": (
        ["The scheduler decides which ready process receives the processor next.",
         "Round robin scheduling gives every process a fixed time quantum in turn.",
         "Shortest job first minimizes average waiting time when burst lengths are known.",
         "Priority scheduling can starve low priority processes unless aging is applied.",
         "A context switch saves the registers of one process and restores another.",
         "Preemptive schedulers interrupt a running process when its quantum expires.",
         "Multilevel feedback queues move processes between queues based on CPU usage.",
         "Turnaround time measures the interval from submission to completion.",
         "Interactive systems favour short response time over raw throughput.",
         "The dispatcher hands control of the CPU to the process chosen by the scheduler."],
        ["how does an operating system choose which process runs on the cpu",
         "round robin time quantum and context switching",
         "preventing starvation in priority scheduling"]),
    "virtual-memory": (
        ["Virtual memory gives each process the illusion of a large private address space.",
         "A page table maps virtual page numbers to physical frame numbers.",
         "A page fault occurs when a referenced page is not resident in memory.",
         "The translation lookaside buffer caches recent address translations.",
         "Least recently used replacement evicts the page unused for the longest time.",
         "Thrashing happens when processes spend more time paging than executing.",
         "Demand paging loads a page only when it is first referenced.",
         "The working set of a process is the set of pages it actively uses.",
         "Copy on write lets parent and child share pages until one of them writes.",
         "Swap space on disk holds pages evicted from physical memory."],
        ["what happens during a page fault",
         "page replacement algorithms like least recently used",
         "why does thrashing slow a computer down"]),
    "photosynthesis": (
        ["Photosynthesis converts light energy into chemical energy stored in glucose.",
         "Chlorophyll in the chloroplast absorbs mostly red and blue light.",
         "The light dependent reactions split water and release oxygen.",
         "The Calvin cycle fixes carbon dioxide into three carbon sugars.",
         "ATP and NADPH from the thylakoid power the Calvin cycle.",
         "Stomata on the leaf surface regulate carbon dioxide intake and water loss.",
         "C4 plants concentrate carbon dioxide to reduce photorespiration.",
         "Rubisco is the enzyme that captures carbon dioxide in the Calvin cycle.",
         "The rate of photosynthesis rises with light intensity until it saturates.",
         "Plants store excess glucose as starch for later use."],
        ["how do plants turn sunlight into sugar",
         "role of chlorophyll and the calvin cycle",
         "why do leaves release oxygen"]),
    "roman-empire": (
        ["Augustus became the first Roman emperor after the fall of the republic.",
         "Roman legions were organized into cohorts and centuries of disciplined soldiers.",
         "The Pax Romana was a long period of relative peace across the empire.",
         "Roman engineers built aqueducts that carried water into growing cities.",
         "The Senate lost real power as emperors centralized authority.",
         "Latin spread through the provinces as the language of administration.",
         "Diocletian divided the empire to make it easier to govern.",
         "The western empire collapsed in 476 when Odoacer deposed the last emperor.",
         "Roman law influenced the legal systems of many later European states.",
         "Constantine moved the capital to Constantinople on the Bosporus."],
        ["who was the first emperor of rome",
         "fall of the western roman empire",
         "roman aqueducts and engineering"]),
    "machine-learning": (
        ["Supervised learning fits a model to labelled input and output pairs.",
         "Gradient descent updates parameters in the direction that reduces the loss.",
         "Overfitting occurs when a model memorizes noise in the training data.",
         "Regularization adds a penalty that discourages overly complex models.",
         "Cross validation estimates how well a model generalizes to unseen data.",
         "The learning rate controls the size of each optimization step.",
         "A validation set is used to tune hyperparameters without touching the test set.",
         "Decision trees split the data on features that best separate the classes.",
         "Ensembles such as random forests average many weak learners.",
         "Precision and recall describe different kinds of classification errors."],
        ["how to prevent overfitting in a model",
         "gradient descent and the learning rate",
         "using cross validation to estimate generalization"]),
    "computer-networks": (
        ["TCP provides reliable ordered delivery using acknowledgements and retransmission.",
         "UDP sends datagrams without connection setup or delivery guarantees.",
         "The three way handshake establishes a TCP connection.",
         "IP routers forward packets based on the destination address prefix.",
         "DNS translates human readable domain names into IP addresses.",
         "Congestion control slows senders when the network drops packets.",
         "The sliding window lets a sender transmit several segments before an ack.",
         "Subnet masks divide an address into network and host portions.",
         "HTTP runs on top of TCP and exchanges requests and responses.",
         "Network address translation lets many hosts share one public address."],
        ["difference between tcp and udp",
         "how does the tcp three way handshake work",
         "what does dns do"]),
    "climate-science": (
        ["Greenhouse gases trap outgoing infrared radiation and warm the surface.",
         "Carbon dioxide concentrations have risen sharply since industrialization.",
         "Melting ice reduces surface albedo and amplifies warming.",
         "Ocean heat content records most of the energy added to the climate system.",
         "Methane is a short lived but potent greenhouse gas.",
         "Climate models simulate the atmosphere, oceans, ice and land together.",
         "Sea level rises as water warms and land ice melts.",
         "Feedback loops can strengthen or weaken an initial temperature change.",
         "Ice cores preserve ancient air bubbles that reveal past climates.",
         "Aerosols reflect sunlight and partly offset greenhouse warming."],
        ["how do greenhouse gases warm the planet",
         "causes of sea level rise",
         "climate feedback loops and albedo"]),
    "cell-biology": (
        ["The cell membrane is a phospholipid bilayer that controls what enters the cell.",
         "Mitochondria produce ATP through cellular respiration.",
         "The nucleus stores DNA and directs protein synthesis.",
         "Ribosomes translate messenger RNA into chains of amino acids.",
         "The endoplasmic reticulum folds and transports newly made proteins.",
         "Mitosis divides one nucleus into two genetically identical nuclei.",
         "Lysosomes contain enzymes that break down waste material.",
         "The cytoskeleton gives the cell its shape and moves organelles.",
         "Active transport uses energy to move molecules against a gradient.",
         "Prokaryotic cells lack a nucleus and membrane bound organelles."],
        ["what do mitochondria do in a cell",
         "how are proteins made by ribosomes",
         "structure of the cell membrane"]),
    "databases": (
        ["A relational database stores data in tables of rows and columns.",
         "SQL queries select, filter, join and aggregate relational data.",
         "An index speeds up lookups at the cost of extra writes and storage.",
         "Transactions follow the ACID properties of atomicity, consistency, isolation and durability.",
         "Normalization removes redundancy by splitting tables into related parts.",
         "A primary key uniquely identifies each row in a table.",
         "Foreign keys enforce relationships between tables.",
         "Write ahead logging lets a database recover after a crash.",
         "Isolation levels trade consistency against concurrency.",
         "Query planners choose join orders and access paths using statistics."],
        ["what are acid transactions",
         "how does a database index speed up queries",
         "database normalization and keys"]),
    "music-theory": (
        ["A major scale follows the pattern of whole and half steps.",
         "Chords are built by stacking intervals of thirds.",
         "The circle of fifths arranges keys by their number of sharps and flats.",
         "Rhythm organizes sound in time using beats and measures.",
         "A time signature tells how many beats fall in each measure.",
         "Minor keys often sound darker than their relative major keys.",
         "Cadences mark the end of musical phrases.",
         "Counterpoint combines independent melodic lines.",
         "Dynamics describe how loudly or softly music is played.",
         "Modulation moves a piece from one key to another."],
        ["how is a major scale constructed",
         "explain the circle of fifths",
         "what is a time signature in music"]),
    "economics": (
        ["Supply and demand together determine the market price of a good.",
         "Inflation is a general rise in prices that erodes purchasing power.",
         "Central banks raise interest rates to slow inflation.",
         "Gross domestic product measures the value of goods and services produced.",
         "Opportunity cost is the value of the next best alternative given up.",
         "Elastic demand changes strongly when price changes.",
         "Fiscal policy uses government spending and taxation to steer the economy.",
         "Comparative advantage explains why countries gain from trade.",
         "Unemployment rises during recessions as output falls.",
         "Monopolies restrict output and raise prices above competitive levels."],
        ["what causes inflation and how do central banks respond",
         "supply and demand price equilibrium",
         "comparative advantage and international trade"]),
    "astronomy": (
        ["Stars form when clouds of gas and dust collapse under gravity.",
         "Nuclear fusion in a star's core converts hydrogen into helium.",
         "A light year is the distance light travels in one year.",
         "Planets orbit the Sun on elliptical paths described by Kepler's laws.",
         "Massive stars end their lives in supernova explosions.",
         "Black holes have gravity so strong that light cannot escape.",
         "Galaxies contain billions of stars bound together by gravity.",
         "Redshift shows that distant galaxies are moving away from us.",
         "White dwarfs are the dense remnants of Sun-like stars.",
         "Exoplanets are detected by transits and stellar wobble."],
        ["how do stars produce energy",
         "what happens when a massive star dies",
         "how are exoplanets discovered"]),
}


@dataclass(frozen=True)
class Document:
    name: str
    topic: str
    text: str


def _paragraph(rng: random.Random, topic: str, label: str) -> str:
    sentences = TOPICS[topic][0]
    chosen = rng.sample(sentences, rng.randint(3, 5))
    return f"Lecture note {label}. " + " ".join(chosen)


def document(seed: int, number: int, topic: str | None = None) -> Document:
    """Return document `number`; the same seed always yields the same text."""
    rng = random.Random(f"{seed}:document:{number}")
    topic = topic or sorted(TOPICS)[rng.randrange(len(TOPICS))]
    paragraphs = [_paragraph(rng, topic, f"{number}.{index}") for index in range(rng.randint(4, 6))]
    return Document(f"doc_{number:05d}.txt", topic, "\n\n".join(paragraphs) + "\n")


def rewrite_paragraph(seed: int, scenario: str, doc: Document) -> Document:
    """Replace one paragraph with fresh text of the same topic (labels unchanged)."""
    rng = random.Random(f"{seed}:{scenario}:edit:{doc.name}")
    paragraphs = doc.text.rstrip("\n").split("\n\n")
    index = rng.randrange(len(paragraphs))
    paragraphs[index] = _paragraph(rng, doc.topic, f"{doc.name}.edit.{index}")
    return Document(doc.name, doc.topic, "\n\n".join(paragraphs) + "\n")


def corpus(seed: int, count: int) -> list[Document]:
    if count < 1:
        raise ValueError("The benchmark corpus needs at least one document")
    # Round robin topic assignment keeps every query's relevant set non-empty.
    topics = sorted(TOPICS)
    return [document(seed, number, topics[number % len(topics)]) for number in range(count)]


def queries() -> list[dict]:
    result = []
    for topic in sorted(TOPICS):
        for number, text in enumerate(TOPICS[topic][1]):
            result.append({"id": f"{topic}-{number}", "topic": topic, "text": text})
    return result


def relevance(documents: list[Document]) -> dict[str, list[str]]:
    """Map each query id to the sorted document paths relevant to it."""
    by_topic: dict[str, list[str]] = {}
    for doc in documents:
        by_topic.setdefault(doc.topic, []).append(f"data/{doc.name}")
    return {query["id"]: sorted(by_topic.get(query["topic"], [])) for query in queries()}


def manifest(seed: int, documents: list[Document]) -> dict:
    files = {doc.name: {"topic": doc.topic, "sha256": hash_bytes(doc.text.encode("utf-8")),
                        "bytes": len(doc.text.encode("utf-8"))} for doc in documents}
    return {"seed": seed, "documents": len(documents), "files": files,
            "corpus_checksum": hash_payload(files), "topics": sorted(TOPICS)}
