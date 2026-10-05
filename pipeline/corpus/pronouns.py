"""Personal, possessive, reflexive, demonstrative and relative pronouns per language: what
counts as a pronominal mention (oracle "pronouns" mode, corpus and text diagnostics)."""

PRONOUNS = {
    "en": set("he him his himself she her hers herself it its itself they them their theirs themselves "
              "this that these those who whom whose which".split()),
    "ru": set("он его него ему нему им ним нём нем она её ее неё нее ей ней ею нею оно они их них им ним "
              "ими ними свой своя своё свое свои своего своей своему своим своих своими себя себе собой "
              "этот эта это эти этого этой этому этим этих этими тот та то те того той тому тем тех теми "
              "который которая которое которые которого которой которому которым которых которыми "
              "котором".split()),
}
