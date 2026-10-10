"""Exercise Lineage mode from file://. It needs no service and sends nothing.

What is checked is the honesty of the answers rather than their look: a
curated edge is never shown as cited, an ambiguous name comes back as a
choice, and a name outside the table returns nothing rather than a neighbour.
"""
from pathlib import Path
import time

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
URL = (ROOT / 'out/index.html').as_uri()


def main():
    output = ROOT / 'build/lineage_checks'
    output.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/usr/bin/google-chrome', headless=True,
                                    args=['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader',
                                          '--enable-unsafe-swiftshader'])
        page = browser.new_page(viewport={'width': 1440, 'height': 1080}, device_scale_factor=1)
        errors, external = [], []
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.on('request', lambda r: external.append(r.url) if r.url.startswith('http') else None)
        page.goto(URL)
        end = time.monotonic() + 30
        while not page.evaluate('() => !!window.LineageExplorer') and time.monotonic() < end:
            page.wait_for_timeout(100)

        page.locator('#mode-lineage').click()
        assert page.locator('#lineage-workspace').is_visible()
        assert not page.locator('.workspace').is_visible()
        assert page.locator('#mode-graph').get_attribute('aria-pressed') == 'false'

        def ask(q):
            page.locator('#q').fill(q)
            page.locator('#query-form button[type=submit]').click()
            return page.locator('#lineage-answer').inner_text()

        # Opens on Karna. Kunti is stated in the text, Surya is the table alone.
        text = page.locator('#lineage-answer').inner_text()
        assert 'parents' in text.lower() and 'Kunti' in text and 'Surya' in text, text
        karna = page.locator('#lineage-answer li', has_text='Surya').first.inner_text()
        assert 'curated, uncited' in karna, karna
        kunti = page.locator('#lineage-answer li', has=page.locator('[data-lineage-node="Kunti"]')).first.inner_text()
        assert 'bornTo' in kunti and 'curated' not in kunti, kunti

        text = ask('Who are the brothers of Arjuna?')
        assert 'Karna' in text and 'Yudhishthira' in text and 'shares Kunti' in text, text

        text = ask('how is karna related to arjuna')
        assert 'a sibling or half-sibling of' in text, text
        text = ask('How is Arjuna related to Parikshit?')
        assert 'Parikshit is a grandchild of Arjuna' in text, text

        # The table has two Chitrasenas and one "Bhimasena (ancestor)". Neither
        # may be chosen for the reader.
        text = ask('Who is Chitrasena?')
        assert 'Which one?' in text and 'Chitrasena (Kaurava)' in text and 'Chitrasena (son of Karna)' in text, text
        text = ask('who is Bhimasena')
        assert 'Which one?' in text, text

        text = ask('Who is Napoleon?')
        assert 'No match' in text, text

        # "arjun" is now caught by the phonetic step, before edit distance.
        text = ask('who is the father of arjun')
        assert 'Read "arjun" as Arjuna by how it sounds' in text and 'Pandu' in text, text
        assert 'does not record which parent is the father' in text

        # Spellings edit distance cannot reach: Bengali o for a, ee for i, j for y.
        text = ask('who is the father of Orjun')
        assert 'Read "Orjun" as Arjuna by how it sounds' in text and 'Pandu' in text, text
        text = ask('who is Bheem')
        assert 'Read "Bheem" as Bhima by how it sounds' in text, text
        text = ask('How is Korno related to Durjodhon?')
        assert 'as Karna' in text and 'as Duryodhana' in text, text
        assert 'Karna (Kaurava) as separate people' in text, text
        # Sound never makes a choice the spelling would not: still a set.
        text = ask('Who is Chitrosen?')
        assert 'Which one?' in text and 'Chitrasena (son of Karna)' in text, text
        # English words whose key collides with a name are left alone.
        text = ask('who was born to Kunti')
        assert 'Barna' not in text, text
        ask('who is the father of arjun')  # back to Arjuna's tree for the click below

        # Selecting a person in the drawing refocuses on them.
        page.locator('#lineage-svg [data-lineage-node="Pandu"]').click()
        assert page.locator('#lineage-focus').inner_text() == 'Pandu'
        page.locator('#lineage-workspace').screenshot(path=str(output / 'tree.png'))

        page.locator('[data-lineage-view=graph]').click()
        drawn = page.locator('#lineage-svg .lnode').count()
        total = page.evaluate('() => LINEAGE.nodes.length')
        assert drawn == total, (drawn, total)
        assert page.locator('#lineage-svg .ledge.is-cited').count() == page.evaluate(
            '() => LINEAGE.edges.filter(e => e.text.length).length')
        page.locator('#lineage-workspace').screenshot(path=str(output / 'graph.png'))

        # Every name in an answer is escaped, so a hostile question stays text.
        ask('<img src=x onerror=alert(1)>')
        assert page.locator('#lineage-answer img').count() == 0

        page.locator('#mode-graph').click()
        assert page.locator('.workspace').is_visible()
        assert not page.locator('#lineage-workspace').is_visible()
        assert page.locator('#mode-lineage').get_attribute('aria-pressed') == 'false'

        assert not errors, errors
        assert not external, 'the page requested something off this machine: %r' % external
        browser.close()
    print('lineage browser checks passed; screenshots in build/lineage_checks')


if __name__ == '__main__':
    main()
