SELECT * FROM expert_data;
SELECT * FROM images;

-- filter to events containing one species
SELECT species, COUNT(*) FROM expert_data WHERE animal_count < 5
GROUP BY species
ORDER BY COUNT(*) DESC;

SELECT url_info FROM images WHERE capture_event_id IN
(SELECT capture_event_id FROM expert_data WHERE species='guineaFowl' AND animal_count < 5);